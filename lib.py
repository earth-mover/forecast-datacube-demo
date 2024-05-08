import itertools
import logging
import random
import string
from dataclasses import dataclass
from typing import Hashable, Iterable

import fsspec
import pandas as pd
import xarray as xr

logger = logging.getLogger("modal_app")
logger.setLevel(logging.INFO)


def random_string(n):
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


@dataclass
class Ingest:
    """Maps a search string to a Zarr group in the store to write to."""

    zarr_group: str
    search: str


@dataclass
class Job:
    runtime: pd.Timestamp
    steps: list[int]
    ingest: Ingest


class ForecastModel:
    def open_herbie(self, job: Job) -> xr.Dataset:
        from herbie import FastHerbie

        FH = FastHerbie(
            DATES=[job.runtime],
            fxx=list(job.steps),
            model="gfs",
        )
        logger.debug("Searching {}".format(job.ingest.search))
        paths = FH.download(search=job.ingest.search)
        logger.debug("Downloaded paths {}".format(paths))  #

        ds = (
            xr.open_mfdataset(sorted(paths), combine="nested", concat_dim="step", engine="cfgrib")
            .expand_dims("time")
            .sortby("step")
        )
        return ds


def batched(iterable, n):
    """From https://docs.python.org/3/library/itertools.html#itertools.batched"""
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def split_into_batches(all_urls: list[list[str]], *, n: int):
    return itertools.chain(*[batched(t, n) for t in all_urls])


def open_single_grib(
    uri: str,
    *,
    expand_dims: Iterable[Hashable] = None,
    drop_vars: Iterable[Hashable] = None,
    chunks="auto",
    **kwargs,
) -> xr.Dataset:
    """Both cfgrib and gribberish require downloading the whole file."""
    ds = xr.open_dataset(
        fsspec.open_local(f"simplecache::{uri}"),
        engine="cfgrib",
        backend_kwargs=kwargs,
        chunks=chunks,
    )
    if drop_vars:
        ds = ds.drop_vars(drop_vars, errors="ignore")
    if expand_dims:
        ds = ds.expand_dims(expand_dims)
    return ds


def get_repo():
    import arraylake as al

    client = al.Client()
    return client.get_or_create_repo("earthmover-demos/gfs")


def create_repo():
    import arraylake as al

    client = al.Client()
    client.delete_repo("earthmover-demos/gfs", imsure=True, imreallysure=True)

    return client.create_repo("earthmover-demos/gfs")
