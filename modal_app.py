# TODO:
# 1. figure out best pattern for shared imports


import itertools
import logging
from datetime import datetime, timedelta
from typing import Any

import modal
import pandas as pd
import xarray as xr
import zarr
from modal import App, Image

from lib import Ingest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

TimestampLike = Any

app = App("example-forecast-ingest")


#############
##### TODO:
GROUP = "avg/"
GRIB_KWARGS = dict(filter_by_keys={"shortName": "prate", "stepType": "avg"})
MAX_URLS = 4
INGEST_GROUPS = [
    Ingest(zarr_group="fcst", search=":(?:PRATE):(?:surface|1000 mb):(?:anl|[0-9]* hour fcst)")
]
#############

MODAL_IMAGE = (
    # Image.micromamba(python_version="3.11")
    Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-dev", "curl")
    .pip_install(
        "arraylake",
        "certifi",
        "cfgrib",
        # "ecmwflibs==0.5.1",
        # "eccodes",
        "dask",
        "fsspec",
        "herbie-data",
        "s3fs",
        "xarray",
        "pydantic-core==2.18.2",
        "pydantic==2.7.1",
        "fastapi>=0.108",
        # channels=["conda-forge"],
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


def backfill(model, *, store, since: TimestampLike, till: TimestampLike | None = None):
    group = GROUP

    write_times(model=model, store=store, since=since, till=till, group=group, mode="w")


def update(model, store):
    import arraylake as al

    # TODO: ???
    group = GROUP

    if isinstance(store, al.repo.ArraylakeStore):
        # fastpath
        instore = store._repo.to_xarray(group)
    else:
        instore = xr.open_zarr(store, group=group)

    since = (
        datetime.combine(instore.time[-1].dt.date.item(), instore.time[-1].dt.time.item())
        + model.update_freq
    )

    logger.info(f"Updating for data since {since}")

    write_times(
        model=model,
        store=store,
        since=since,
        till=None,
        group=group,
        mode="a-",
        append_dim="time",
    )


def write_times(*, model, store, since, till=None, **write_kwargs):
    import arraylake as al

    import lib

    available_times = model.get_available_times(since, till)
    logger.info(f"Available times are {available_times}")

    schema = model.create_schema(times=available_times)
    logger.info(f"schema {schema}")

    logger.info("Writing schema")
    schema.to_zarr(store, **write_kwargs, compute=False)

    # figure out total number of timestamps in store.
    ntimes = zarr.open_group(store)[f"{GROUP}/time"].size

    # TODO: not really needed?
    # if isinstance(store, al.repo.ArraylakeStore):
    #    store._repo.commit(f"Initialized for update since {since!r}")

    # TODO: loop over vars? Or set this some other way.
    chunksizes = dict(zip(schema.prate.dims, schema.prate.encoding["chunks"]))

    logger.info("Starting write job.")
    all_jobs = (
        lib.Job(runtime=time, steps=steps, ingest=ingest_group)
        for time, steps, ingest_group in itertools.product(
            available_times, lib.batched(model.step, n=chunksizes["step"]), INGEST_GROUPS
        )
    )

    list(
        write_herbie.map(
            all_jobs,
            kwargs={
                "model": model,
                "group": write_kwargs.get("group", None),
                "schema": schema,
                "store": store,
                "ntimes": ntimes,
            },
        )
    )

    logger.info("Finished write job.")
    if isinstance(store, al.repo.ArraylakeStore):
        since = pd.Timestamp(since).floor(model.update_freq)
        store._repo.commit(f"Finished update since {since!r}")


@app.function(**MODAL_FUNCTION_KWARGS, timeout=900)
def write_herbie(job, *, model, group, schema, store, ntimes=None):
    import time

    import numpy as np

    tic = time.time()

    logger.debug("Processing job {}".format(job))
    # chunksizes = dict(zip(schema.prate.dims, schema.prate.encoding["chunks"]))
    try:
        ds = (
            model.open_herbie(job)
            .chunk(step=-1)
            # TODO: transpose_like
            .transpose("time", "step", "latitude", "longitude")
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
            "latitude": slice(None),
            "longitude": slice(None),
            "step": slice(istep[0], istep[-1] + 1),
            "time": time_region,
        }

        logger.info(f"Writing job {job} to region {region}")

        # Drop coordinates to avoid useless overwriting
        # Verified that this only writes data_vars array chunks
        ds.drop_vars(ds.coords).to_zarr(store, group=group, region=region)
    except Exception as e:
        raise RuntimeError(f"Failed for {job}") from e

    logger.info("Finished job {}. Took {} sconds".format(job, time.time() - tic))
    return ds.step


@app.function(**MODAL_FUNCTION_KWARGS, timeout=900)
def write_uri(uri, *, model, group, schema, store, ntimes=None):
    import numpy as np

    chunksizes = dict(zip(schema.prate.dims, schema.prate.encoding["chunks"]))
    logger.info(f"Writing uri {uri}")
    try:
        ds = (
            model.open_multiple_gribs(uri, **GRIB_KWARGS)
            .chunk(step=chunksizes["step"])
            .transpose("time", "step", "latitude", "longitude")
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
        assert (np.diff(istep) == 1).all()

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
            "latitude": slice(None),
            "longitude": slice(None),
            "step": slice(istep[0], istep[-1] + 1),
            "time": time_region,
        }

        logger.info(f"Writing uri {uri} to region {region}")

        # Drop coordinates to avoid useless overwriting
        # Verified that this only writes data_vars array chunks
        ds.drop_vars(ds.coords).to_zarr(store, group=group, region=region)
    except Exception as e:
        raise RuntimeError(f"Failed for {uri}") from e

    logger.info(f"Finished uri {uri}")
    return ds.step


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600)
def gfs_ingest():
    import gfs
    import lib

    # print("This code is running on a remote machine.")
    GFS = gfs.GFS()

    # import pydantic
    # print(pydantic.__version__)
    # import arraylake as al
    # print(al.diagnostics.get_versions())

    repo = lib.create_repo()
    store = repo.store

    backfill(
        GFS,
        store=store,
        since=datetime.utcnow() - timedelta(days=2),
        till=datetime.utcnow() - timedelta(days=1, hours=12),
    )


@app.local_entrypoint()
def main():
    gfs_ingest.remote()
    # print(check_httpx.remote())


# @app.function(image=IMAGE)
# def check_httpx():
#    import httpx

# import certifi
#    return httpx.get("https://api.earthmover.io").json()
# return certifi.where()
# print(os.environ["SSL_CERT_DIR"])
# return os.listdir(os.environ["SSL_CERT_DIR"])
