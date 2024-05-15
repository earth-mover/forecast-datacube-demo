import itertools
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import arraylake as al
import modal
import numpy as np
import pandas as pd
import tomllib
import xarray as xr
import zarr
from modal import App, Image

from src import lib, models
from src.lib import Ingest, get_logger

logger = get_logger()
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

TimestampLike = Any

app = App("forecast-ingest-lib")


MODAL_IMAGE = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-dev", "curl")
    .pip_install(
        "arraylake",
        "certifi",
        "cfgrib",
        "dask",
        "fsspec",
        "herbie-data",
        "s3fs",
        "xarray",
        "pydantic-core==2.18.2",
        "pydantic==2.7.1",
        "fastapi>=0.108",
    )
    .pip_install("eccodes", "ecmwflibs")
    .env(
        {
            "SSL_CERT_FILE": "/opt/conda/lib/python3.11/site-packages/certifi/cacert.pem",  # noqa: E501
        }
    )
    .run_commands("python -m eccodes selfcheck")
)
MODAL_FUNCTION_KWARGS = dict(
    image=MODAL_IMAGE,
    secrets=[
        modal.Secret.from_name("earth-mover-aws-secret"),
        modal.Secret.from_name("deepak-arraylake-demos-token"),
    ],
    mounts=[modal.Mount.from_local_python_packages("src")],
    #    concurrency_limit=1,
)


#############


@app.function(**MODAL_FUNCTION_KWARGS)
def initialize(ingest) -> None:
    logger.info("initialize: for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    group = ingest.zarr_group
    store = ingest.zarr_store

    if ingest.chunks[model.runtime_dim] != 1:
        raise NotImplementedError(
            "Chunk size along the {model.runtime_dim!r} dimensions must be 1."
        )

    # We write the schema for time-invariant variables first to prevent conflicts
    schema = (
        model.create_schema(chunksizes=ingest.chunks, search=ingest.search, renames=ingest.renames)
        .coords.to_dataset()
        .drop_dims([model.runtime_dim])
    )
    schema.to_zarr(store, group=group, mode="w")

    if isinstance(store, al.repo.ArraylakeStore):
        store._repo.commit("Write schema for backfill ingest: {}".format(ingest))


@app.function(**MODAL_FUNCTION_KWARGS)
def backfill(
    ingest: Ingest,
    *,
    since: TimestampLike,
    till: TimestampLike | None = None,
) -> None:
    logger.info("backfill: Running for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    if till is not None:
        till = till + model.update_freq
    write_times(since=since, till=till, ingest=ingest, initialize=True, mode="a")


@app.function(**MODAL_FUNCTION_KWARGS)
def update(ingest: Ingest) -> None:
    logger.info("update: Running for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    group = ingest.zarr_group
    store = ingest.zarr_store
    assert store is not None

    if isinstance(store, al.repo.ArraylakeStore):
        # fastpath
        instore = store._repo.to_xarray(group)
    else:
        instore = xr.open_zarr(store, group=group)

    latest_available_date = model.latest_available(ingest)
    latest_in_store = pd.Timestamp(instore.time[slice(-1, None)].data[0])
    if latest_in_store == latest_available_date:
        # already ingested
        logger.info(
            "No new data. Latest data {}; Latest in-store: {}".format(
                latest_available_date, latest_in_store
            )
        )
        return

    since = (
        datetime.combine(instore.time[-1].dt.date.item(), instore.time[-1].dt.time.item())
        + model.update_freq
    )

    # Our interval is left-closed [since, till) but latest_available_date must be included
    # So adjust `till`
    till = latest_available_date + model.update_freq
    logger.info("Latest data not complete but there is data to ingest. ")

    logger.info(
        "update: There is data to ingest. "
        "Latest data {}; Latest in-store: {}. "
        "Calling write_times for data since {} till {}".format(
            latest_available_date, latest_in_store, since, till
        )
    )
    write_times(since=since, till=till, mode="a-", append_dim=model.runtime_dim, ingest=ingest)


def write_times(
    *,
    since: pd.Timestamp,
    till: pd.Timestamp | None,
    ingest: Ingest,
    initialize: bool = False,
    **write_kwargs,
) -> None:
    """
    Writes data for a range of times to the store.

    First initializes (if ``mode="w"`` in ``write_kwargs``) or resizes the store
    (if ``mode="a"`` in ``write_kwargs``) by writing a "schema" dataset of the right
    shape. Then begins a distributed region write to that new section of the dataset.

    Parameters
    ----------
    model: ForecastModel
    store:
        Zarr store
    since: Timestamp
       Anything that can be cast to a pandas.Timestamp
    till: Timestamp
       Anything that can be cast to a pandas.Timestamp
    ingest: Ingest
       Contains  ``search`` string for variables to write to the
       specified group ``zarr_group`` of the ``store``.
    **write_kwargs:
       Extra kwargs for writing the schema.
    """
    group = ingest.zarr_group
    store = ingest.zarr_store
    model = models.get_model(ingest.model)
    assert store is not None

    available_times = model.get_available_times(since, till)
    logger.info("Available times are {} for ingest {}".format(available_times, ingest))

    schema = model.create_schema(
        times=available_times,
        search=ingest.search,
        chunksizes=ingest.chunks,
        renames=ingest.renames,
    )
    # Drop time-invariant variables to prevent conflicts
    to_drop = [name for name, var in schema.variables.items() if model.runtime_dim not in var.dims]

    if initialize:
        # Initialize with the right attributes. We do not update these after initialization
        logger.info("Getting attributes for data variables.")
        dset = model.as_xarray(ingest.search).expand_dims(model.expand_dims)
        if sorted(schema.data_vars) != sorted(dset.data_vars):
            raise ValueError(
                "Please add or update the `renames` field for this job in the TOML file. "
                f"Constructed schema expects data_vars={tuple(schema.data_vars)!r}. "
                f"Loaded data has data_vars={tuple(dset.data_vars)!r}"
            )
        for name, var in dset.data_vars.items():
            schema[name].attrs = var.attrs

    # 1. figure out total number of timestamps in store.
    zarr_group = zarr.open_group(store)
    ntimes = zarr_group[f"{group.removesuffix('/')}/time"].size

    # Workaround for Xarray overwriting group attrs.
    # https://github.com/pydata/xarray/issues/8755
    schema.attrs.update(zarr_group.attrs.asdict())

    logger.info("Writing schema: {}".format(schema))
    schema.drop_vars(to_drop).to_zarr(store, group=group, **write_kwargs, compute=False)

    step_hours = (schema.indexes["step"].asi8 / 1e9 / 3600).astype(int).tolist()

    if ingest.chunks[model.runtime_dim] != 1:
        raise NotImplementedError

    time_and_steps = itertools.chain(
        *(
            itertools.product(
                (t,),
                lib.batched(
                    (step for step in model.get_steps(t) if step in step_hours),
                    n=ingest.chunks[model.step_dim],
                ),
            )
            for t in available_times
        )
    )

    logger.info("Starting write job for {}.".format(ingest.search))
    all_jobs = (
        lib.Job(runtime=time, steps=steps, ingest=ingest)
        for ingest, (time, steps) in itertools.product([ingest], time_and_steps)
    )

    list(write_herbie.map(all_jobs, kwargs={"schema": schema, "ntimes": ntimes}))

    logger.info("Finished write job for {}.".format(ingest))
    if isinstance(store, al.repo.ArraylakeStore):
        store._repo.commit(
            f"""
            Finished update: {available_times[0]!r}, till {available_times[-1]!r}.\n
            Data: {ingest.model}, {ingest.product}, {ingest.search}.\n
            zarr_group: {ingest.zarr_group}
            """
        )


@app.function(**MODAL_FUNCTION_KWARGS, timeout=240, retries=3)
def write_herbie(job, *, schema, ntimes=None):
    tic = time.time()

    ingest = job.ingest
    model = models.get_model(ingest.model)
    store = ingest.zarr_store
    group = ingest.zarr_group
    assert store is not None

    logger.debug("Processing job {}".format(job))
    try:
        ds = model.open_herbie(job)

        ##############################
        ###### manually get region to avoid each task reading in the same vector
        # We have optimized so that
        # (1) length of the time dimension is passed in to each worker.
        # (2) The timestamps we are writing are passed in `schema`
        # (3) we know the `step` values.
        # So we can infer `region` without making multiple trips to the store.
        index = pd.to_timedelta(schema.indexes["step"], unit="hours")
        step = pd.to_timedelta(ds.step.data, unit="ns")
        istep = index.get_indexer(step)
        if (istep == -1).any():
            raise ValueError(f"Could not find all of step={ds.step.data!r} in {index!r}")
        if not (np.diff(istep) == 1).all():
            raise ValueError(f"step is not continuous={ds.step.data!r}")

        if ntimes is None:
            time_region = "auto"
        else:
            index = schema.indexes["time"]
            itime = index.get_indexer(ds.time.data).item()
            if itime == -1:
                raise ValueError(f"Could not find time={ds.time.data!r} in {index!r}")
            ntimes -= len(index)  # existing number of timestamps
            time_region = slice(ntimes + itime, ntimes + itime + 1)
        #############################

        region = {
            "step": slice(istep[0], istep[-1] + 1),
            "time": time_region,
        }
        region.update({dim: slice(None) for dim in model.dim_order if dim not in region})

        logger.info("Writing job {} to region {}".format(job, region))

        # Drop coordinates to avoid useless overwriting
        # Verified that this only writes data_vars array chunks
        ds.drop_vars(ds.coords).to_zarr(store, group=group, region=region)
    except Exception as e:
        raise RuntimeError(f"Failed for {job}") from e

    logger.info("Finished writing job {}. Took {} seconds".format(job, time.time() - tic))
    return ds.step


def parse_toml_config(file: str) -> dict[str, lib.Ingest]:
    with open(file, mode="rb") as f:
        parsed = tomllib.load(f)

    ingest_jobs = defaultdict(list)
    for key, values in parsed.items():
        searches = values.pop("searches")
        for search in searches:
            ingest_jobs[key].append(lib.Ingest(name=key, **values, search=search))
    return ingest_jobs


def driver(*, mode, ingest_jobs, since=None, till=None):
    # Set this here for Arraylake so all tasks start with the same state
    for v in ingest_jobs.values():
        for i in v:
            i.zarr_store = lib.get_zarr_store(i.store)

    ingests = itertools.chain(*ingest_jobs.values())
    if mode == "backfill":
        # TODO: assert zarr_store/group is not duplicated
        for_init = tuple(next(iter(v)) for v in ingest_jobs.values())
        list(initialize.map(for_init))

        # initialize commits the schema, so we need to checkout again
        for v in ingest_jobs.values():
            for i in v:
                i.zarr_store = lib.get_zarr_store(i.store)

        list(backfill.map(ingests, kwargs={"since": since, "till": till}))

    elif mode == "update":
        list(update.map(ingests))

    else:
        raise NotImplementedError()


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600)
def hrrr_backfill():
    file = "src/configs/hrrr.toml"
    mode = "backfill"  # "update", or "backfill"
    since = datetime.utcnow() - timedelta(days=3)
    till = datetime.utcnow() - timedelta(days=1, hours=12)

    ingest_jobs = parse_toml_config(file)

    driver(mode=mode, ingest_jobs=ingest_jobs, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600, schedule=modal.Cron("57 * * * *"))
def hrrr_update_solar():
    file = "src/configs/hrrr.toml"
    mode = "update"  # "update", or "insert"

    ingest_jobs = parse_toml_config(file)
    driver(mode=mode, ingest_jobs=ingest_jobs)


@app.local_entrypoint()
def main():
    hrrr_backfill.remote()
    # Command-line kwargs
    # modal_mode = "run"

    # # In config TOML file
    # name = "hrrr_update"
    # cron = "30 * * * *"

    # modal_kwargs = dict(name=name, **MODAL_FUNCTION_KWARGS, timeout=3600)
    # # if modal_mode == "deploy":
    # #     if cron:
    # #         modal_kwargs["schedule"] = modal.Cron("30 * * * *")
    # #     driver_function = app.function(**modal_kwargs)(interface)
    # #     modal.runner.deploy_app(app, name=name)
    # # else:
    # driver_function = app.function(**modal_kwargs)(interface)
    # driver_function.remote()
