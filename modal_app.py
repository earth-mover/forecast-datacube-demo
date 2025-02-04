import itertools
import logging
import random
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import arraylake as al
import icechunk
import modal
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from src import lib, models
from src.lib import Ingest, ReadMode, WriteMode, get_logger, merge_searches
from src.lib_modal import MODAL_FUNCTION_KWARGS

logger = get_logger()
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

TimestampLike = Any

LOGGING_STORE: bool = True  # TODO: make a commandline flag / env var

applib = modal.App("earthmover-forecast-ingest-lib")


def maybe_commit(store, message) -> None:
    if isinstance(store, al.repo.ArraylakeStore):
        store._repo.commit(message)
    elif isinstance(store, icechunk.Session):
        store.commit(message)
    elif isinstance(store, icechunk.IcechunkStore):
        store.session.commit(message)


#############
def get_store(uri):
    import os

    import arraylake as al
    from arraylake import Client

    client = Client(token=os.environ["ARRAYLAKE_TOKEN"])
    repo = lib.maybe_get_repo(uri, client=client)
    if isinstance(repo, al.repo.Repo):
        return repo.store
    elif isinstance(repo, icechunk.Repository):
        # TODO: add imperative `pickle preserves read_only method`
        #       a context manager won't work here
        return repo.writable_session("main").store
    else:
        return repo


def to_zarr_kwargs(store) -> dict[str, Any]:
    if isinstance(store, icechunk.IcechunkStore):
        return {"zarr_format": 3, "consolidated": False}
    return {}


def initialize(ingest) -> None:
    """This function initializes a Zarr group with the schema for a backfill."""
    logger.info("initialize: for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    group = ingest.zarr_group
    store = get_store(ingest.store)

    if ingest.chunks[model.runtime_dim] != 1:
        raise NotImplementedError(
            "Chunk size along the {model.runtime_dim!r} dimensions must be 1."
        )

    # We write the schema for time-invariant variables first to prevent conflicts
    schema = model.create_schema(ingest).coords.to_dataset().drop_dims([model.runtime_dim])
    logger.debug(f"Writing schema to {group=!r} with extra kwargs: {to_zarr_kwargs(store)!r}")
    schema.to_zarr(store, group=group, mode="w", **to_zarr_kwargs(store))

    maybe_commit(store, "Write schema for backfill ingest: {}".format(ingest))


@applib.function(**MODAL_FUNCTION_KWARGS, timeout=1200)
def verify(ingest: Ingest, *, nsteps=None):
    import dask

    tic = time.time()
    store = get_store(ingest.store)
    inrepo = xr.open_dataset(
        store, group=ingest.zarr_group, chunks=None, engine="zarr", consolidated=False
    )
    model = models.get_model(ingest.model)
    timedim, stepdim = model.runtime_dim, model.step_dim

    runtime = pd.Timestamp(random.choice(inrepo[timedim].data))
    step = model.get_steps(runtime)
    if nsteps is not None:
        step = sorted(random.sample(step, k=nsteps))
    job = lib.Job(runtime=runtime, steps=step, ingest=ingest)
    logger.info("verify: Running for job: {}".format(job))

    actual = inrepo.sel({timedim: [runtime], stepdim: [pd.Timedelta(s, unit="h") for s in step]})
    expected = model.open_herbie(job)
    # manage our own pool so we can shutdown intentionally
    pool = ThreadPoolExecutor()
    try:
        # attributes are different, so identical won't work.
        # load now, otherwise Xarray loads in a for-loop
        with dask.config.set(pool=pool):
            xr.testing.assert_allclose(
                lib.clean_dataset(actual, model).chunk().load(),
                lib.clean_dataset(expected, model).load(),
            )
        logger.info("Successfully verified!")
    except Exception as e:
        raise RuntimeError(f"Verify failed for {job}") from e
    finally:
        pool.shutdown()
    logger.info("Finished verifying job {}. Took {} seconds".format(job, time.time() - tic))


@applib.function(**MODAL_FUNCTION_KWARGS, timeout=3600 * 3)
def backfill(
    ingest: Ingest,
    *,
    since: TimestampLike,
    till: TimestampLike | None = None,
) -> None:
    """This function runs after `initialize` and sets up the `write_times` function."""
    logger.info("backfill: Running for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    if till is not None:
        till = till + model.update_freq
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="This pattern is interpreted as a regular expression*",
        )
    initialize(ingest)
    write_times(since=since, till=till, ingest=ingest, initialize=True, mode="a")


@applib.function(**MODAL_FUNCTION_KWARGS, timeout=3600 * 3)
def update(ingest: Ingest) -> None:
    """This function sets up the `write_times` function for a new update to the dataset."""
    logger.info("update: Running for ingest {}".format(ingest))
    model = models.get_model(ingest.model)
    group = ingest.zarr_group
    store = get_store(ingest.store)
    assert store is not None

    if isinstance(store, al.repo.ArraylakeStore):
        # fastpath
        instore = store._repo.to_xarray(group)
    elif isinstance(store, icechunk.IcechunkStore):
        instore = xr.open_zarr(
            store, group=group, zarr_format=3, consolidated=False, decode_timedelta=True
        )
    else:
        instore = xr.open_zarr(store, group=group, mode="r", decode_timedelta=True)

    latest_available_date = model.latest_available(ingest)
    time = instore[model.runtime_dim]
    latest_in_store = pd.Timestamp(time[slice(-1, None)].data[0])
    if latest_in_store == latest_available_date:
        # already ingested
        logger.info(
            "No new data. Latest data {}; Latest in-store: {}".format(
                latest_available_date, latest_in_store
            )
        )
        return

    since: pd.Timestamp = pd.Timestamp(
        datetime.combine(time[-1].dt.date.item(), time[-1].dt.time.item()) + model.update_freq
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

    This is an orchestrator that cuts up the job in to chunks and sends them on.
    The actual writing happens in the spawned ``write_herbie`` functions.

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
    store = get_store(ingest.store)
    if isinstance(store, icechunk.IcechunkStore):
        session = store.session  # this is a new session :/
    model = models.get_model(ingest.model)
    assert store is not None

    available_times = model.get_available_times(since, till)
    logger.info("Available times are {} for ingest {}".format(available_times, ingest))

    schema = model.create_schema(times=available_times, ingest=ingest)
    # Drop time-invariant variables to prevent conflicts
    to_drop = [name for name, var in schema.variables.items() if model.runtime_dim not in var.dims]

    if initialize:
        # Initialize with the right attributes. We do not update these after initialization
        # This is a little ugly, but it minimizes code duplication.
        logger.info("Getting attributes for data variables.")
        dset = model.as_xarray(merge_searches(ingest.searches)).expand_dims(model.expand_dims)
        if sorted(schema.data_vars) != sorted(dset.data_vars):
            raise ValueError(
                "Please add or update the `renames` field for this job in the TOML file. "
                f"Constructed schema expects data_vars={tuple(schema.data_vars)!r}. "
                f"Loaded data has data_vars={tuple(dset.data_vars)!r}"
            )
        for name, var in dset.data_vars.items():
            schema[name].attrs = var.attrs

    zarr_group = zarr.open_group(store)[group]

    # Workaround for Xarray overwriting group attrs.
    # https://github.com/pydata/xarray/issues/8755
    schema.attrs.update(zarr_group.attrs.asdict())

    logger.info("Writing schema for initialize: {}".format(schema))
    # TODO: remove when zarr v3 supports write_empty_chunks
    more_kwargs = to_zarr_kwargs(store)
    if more_kwargs.get("zarr_format") == 3:
        for var in schema._variables.values():
            var.encoding.pop("write_empty_chunks", None)
            var.encoding.pop("compressor", None)
            var.encoding.pop("filters", None)
            # if codecs := var.encoding.pop("filters", None):
            #     comp = var.encoding.pop("compressor", None)
            #     var.encoding["codecs"] = codecs + (comp,) if comp is not None else ()
    schema.drop_vars(to_drop).to_zarr(
        store, group=group, **write_kwargs, compute=False, **more_kwargs
    )

    step_hours = (schema.indexes["step"].asi8 / 1e9 / 3600).astype(int).tolist()

    if ingest.chunks[model.runtime_dim] != 1:
        raise NotImplementedError

    with session.allow_pickling():  # FIXME: assumes icechunk
        ingest.zarr_store = session.store
        # TODO: This is the place to update if we wanted chunksize along `model.runtime_dim`
        # to be greater than 1.
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

        logger.info("Starting write job for {}.".format(ingest.searches))
        all_jobs = (
            lib.Job(runtime=time, steps=steps, ingest=ingest)
            for ingest, (time, steps) in itertools.product(ingest, time_and_steps)
        )

        # figure out total number of timestamps in store.
        # This is an optimization to figure out the `region` in `write_herbie`
        # minimizing number of roundtrips to the object store.
        ntimes = zarr_group[model.runtime_dim].size
        results = list(write_herbie.map(all_jobs, kwargs={"schema": schema, "ntimes": ntimes}))

    logger.info("Finished write job for {}.".format(ingest))
    message = f"""
            Finished update: {available_times[0]!r}, till {available_times[-1]!r}.\n
            Data: {ingest.model}, {ingest.product} \n
            Searches: {ingest.searches}.\n
            zarr_group: {ingest.zarr_group}
            """
    if isinstance(store, icechunk.IcechunkStore):
        logger.info("Merging changesets for icechunk")
        for _, next_store in results:
            session.merge(next_store.session)
        maybe_commit(session, message)
    else:
        maybe_commit(store, message)


@applib.function(**MODAL_FUNCTION_KWARGS, timeout=300, retries=10)
def write_herbie(job, *, schema, ntimes=None):
    """Actual writes data to disk."""
    import dask

    # manage our own pool so we can shutdown intentionally
    pool = ThreadPoolExecutor(4)

    tic = time.time()

    ingest = job.ingest
    model = models.get_model(ingest.model)
    store = ingest.zarr_store
    group = ingest.zarr_group

    logger.debug("Processing job {}".format(job))
    try:
        ds = model.open_herbie(job)

        ##############################
        ###### manually get region to avoid each task reading in the same vector
        # We have optimized so that
        # (1) length of the time dimension is passed in to each worker.
        # (2) The timestamps we are writing are passed in `schema`
        # (3) we know the `step` values.
        # So we can infer `region` without making multiple trips to the Zarr store.
        index = pd.to_timedelta(schema.indexes[model.step_dim], unit="hours")
        step_data = ds[model.step_dim].data
        step = pd.to_timedelta(step_data, unit="ns")
        istep = index.get_indexer(step)
        if (istep == -1).any():
            raise ValueError(f"Could not find all of step={step_data!r} in {index!r}")
        if not (np.diff(istep) == 1).all():
            raise ValueError(f"step is not continuous={step_data!r}")

        if ntimes is None:
            time_region = "auto"
        else:
            index = schema.indexes[model.runtime_dim]
            itime = index.get_indexer(ds[model.runtime_dim].data).item()
            if itime == -1:
                raise ValueError(f"Could not find time={ds.time.data!r} in {index!r}")
            ntimes -= len(index)  # existing number of timestamps
            time_region = slice(ntimes + itime, ntimes + itime + 1)
        #############################

        region = {
            model.step_dim: slice(istep[0], istep[-1] + 1),
            model.runtime_dim: time_region,
        }
        region.update(
            {dim: slice(None) for dim in model.dim_order if dim not in region and dim in ds.dims}
        )

        logger.info("Writing job {} to region {}".format(job, region))

        # Drop coordinates to avoid useless overwriting
        # Verified that this only writes data_vars array chunks
        with dask.config.set(pool=pool):
            loaded = ds.drop_vars(ds.coords).compute()
            logger.info(
                "      loaded data for job {}. Took {} seconds since start".format(
                    job.summarize(), time.time() - tic
                )
            )
            if LOGGING_STORE:
                from zarr.storage import LoggingStore

                store_ = LoggingStore(store)
            else:
                store_ = store
            with zarr.config.set({"async.concurrency": 1}):
                loaded.to_zarr(store_, group=group, region=region, **to_zarr_kwargs(store))
    except Exception as e:
        raise RuntimeError(f"Failed for {job}") from e
    finally:
        logger.info(
            "Shutting down pool for job {}. Took {} seconds".format(
                job.summarize(), time.time() - tic
            )
        )
        pool.shutdown()

    logger.info(
        "      Finished writing job {}. Took {} seconds".format(job.summarize(), time.time() - tic)
    )
    return (ds.step.data, store)


def driver(*, mode: WriteMode | ReadMode, toml_file_path: str, since=None, till=None) -> None:
    """Simply dispatches between update and backfill modes."""
    ingest_jobs = lib.parse_toml_config(toml_file_path)

    env = subprocess.run(["pip", "list"], capture_output=True, text=True)
    logger.info(env.stdout)

    # Set this here for Arraylake so all tasks start with the same state
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The value of the smallest subnormal"
        )
        ingests = ingest_jobs.values()
        for i in ingests:
            # create the store if needed
            lib.maybe_get_repo(i.store)

    if mode is WriteMode.BACKFILL:
        # TODO: assert zarr_store/group is not duplicated
        list(backfill.map(ingests, kwargs={"since": since, "till": till}))

    elif mode is WriteMode.UPDATE:
        list(update.map(ingests))

    elif mode is ReadMode.VERIFY:
        list(verify.map(ingests))

    else:
        raise NotImplementedError
