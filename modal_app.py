# TODO:
# 1. figure out best pattern for shared imports
# 2. Allow adding new variables to existing group with mode="a".
# 3. Schema / chunksize setting


import itertools
import logging
from datetime import datetime, timedelta
from typing import Any

import modal
import pandas as pd
import tenacity
import xarray as xr
import zarr
from modal import App, Image

from lib import ForecastModel, Ingest, get_logger

logger = get_logger()
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

TimestampLike = Any

app = App("example-forecast-ingest")


#############
##### TODO:
INGEST_GROUPS = [
    # Ingest(
    #    zarr_group="avg/",
    #    search=":(?:PRATE|GRD):(?:surface|1000 mb):(?:anl|[0-9]* hour fcst)",
    # ),
    Ingest(
        product="sfc",
        zarr_group="fcst/",
        search="DSWRF:surface:(?=anl|[0-9]* hour fcst)",
    ),
]
#############

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
        "tenacity",
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
        modal.Secret.from_name("deepak-aws-secret"),
        modal.Secret.from_name("deepak-arraylake-demos-token"),
    ],
    mounts=[modal.Mount.from_local_python_packages("lib", "gfs")],
)


#############


@app.function(**MODAL_FUNCTION_KWARGS)
def backfill(
    ingest: Ingest,
    *,
    model: ForecastModel,
    store,
    since: TimestampLike,
    till: TimestampLike | None = None,
):
    logger.info("backfill: Calling write_times for ingest {}".format(ingest))
    write_times(model=model, store=store, since=since, till=till, ingest=ingest, mode="w")


@app.function(**MODAL_FUNCTION_KWARGS)
def update(ingest: Ingest, model: ForecastModel, store):
    import arraylake as al

    group = ingest.zarr_group

    if isinstance(store, al.repo.ArraylakeStore):
        # fastpath
        instore = store._repo.to_xarray(group)
    else:
        instore = xr.open_zarr(store, group=group)

    latest = model.latest()
    if (
        # already ingested
        pd.Timestamp(instore.time[slice(-1, None)].data[0]) == latest.date
        # data not ready yet
        # TODO: this probably needs to be better
        or len(latest.inventory(ingest.search)) == 1
    ):
        logger.info("No new data.")
        return

    since = (
        datetime.combine(instore.time[-1].dt.date.item(), instore.time[-1].dt.time.item())
        + model.update_freq
    )

    logger.info("update: Calling write_times for data since {}".format(since))

    write_times(
        model=model,
        store=store,
        since=since,
        till=None,
        mode="a-",
        append_dim=model.runtime_dim,
        ingest=ingest,
    )


def write_times(*, model, store, since, till=None, ingest: Ingest, **write_kwargs):
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
    import arraylake as al

    import lib

    group = ingest.zarr_group

    available_times = model.get_available_times(since, till)
    logger.info("Available times are {} for ingest {}".format(available_times, ingest))

    schema = model.create_schema(times=available_times, search=ingest.search)
    logger.info(f"schema {schema}")

    logger.info("Writing schema")
    schema.to_zarr(store, group=group, **write_kwargs, compute=False)

    # figure out total number of timestamps in store.
    ntimes = zarr.open_group(store)[f"{group}/time"].size

    # TODO: set this some other way
    var = next(iter(schema.data_vars))
    chunksizes = dict(zip(schema[var].dims, schema[var].encoding["chunks"]))

    time_and_steps = itertools.chain(
        *(
            itertools.product(
                (t,), lib.batched(model.get_steps(t - t.floor("D")), n=chunksizes[model.step_dim])
            )
            for t in available_times
        )
    )

    logger.info("Starting write job for {}.".format(ingest.search))
    all_jobs = (
        lib.Job(runtime=time, steps=steps, ingest=ingest)
        for ingest, (time, steps) in itertools.product([ingest], time_and_steps)
    )

    list(
        write_herbie.map(
            all_jobs,
            kwargs={
                "model": model,
                "schema": schema,
                "store": store,
                "ntimes": ntimes,
            },
        )
    )

    logger.info("Finished write job for {}.".format(ingest))
    if isinstance(store, al.repo.ArraylakeStore):
        store._repo.commit(
            f"Finished update since {available_times[0]!r}, till {available_times[-1]!r}."
        )


@app.function(**MODAL_FUNCTION_KWARGS, timeout=900)
@tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(3))
def write_herbie(job, *, model, schema, store, ntimes=None):
    import time

    import numpy as np

    tic = time.time()

    logger.debug("Processing job {}".format(job))
    try:
        ds = (
            model.open_herbie(job)
            # We *should* only be processing in a single Zarr group chunksize
            # number of steps
            .chunk(step=-1)
            # TODO: transpose_like(schema)
            .transpose(*model.dim_order)
        )

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
        ds.drop_vars(ds.coords).to_zarr(store, group=job.ingest.zarr_group, region=region)
    except Exception as e:
        raise RuntimeError(f"Failed for {job}") from e

    logger.info("Finished job {}. Took {} sconds".format(job, time.time() - tic))
    return ds.step


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600)
def ingest():
    import gfs
    import lib

    # print("This code is running on a remote machine.")
    # GFS = gfs.GFS()
    model = gfs.HRRR()

    # import pydantic
    # print(pydantic.__version__)
    # import arraylake as al
    # print(al.diagnostics.get_versions())

    repo = lib.create_repo(model.name)
    store = repo.store
    # breakpoint()
    list(
        backfill.map(
            INGEST_GROUPS,
            kwargs={
                "model": model,
                "store": store,
                "since": datetime.utcnow() - timedelta(days=3),
                "till": datetime.utcnow() - timedelta(days=1, hours=12),
            },
        )
    )

    # repo = lib.get_repo(model.name)
    # store = repo.store
    # list(update.map(INGEST_GROUPS, kwargs={"model": model, "store": store}))


@app.local_entrypoint()
def main():
    ingest.remote()
