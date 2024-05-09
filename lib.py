import contextlib
import itertools
import logging
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Hashable, Iterable

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr


def get_logger():
    logger = logging.getLogger("modal_app")
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def random_string(n):
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


@dataclass
class Ingest:
    """Maps a search string to a Zarr group in the store to write to."""

    product: str
    zarr_group: str
    search: str


@dataclass
class Job:
    runtime: pd.Timestamp
    steps: list[int]
    ingest: Ingest


class ForecastModel(ABC):
    @abstractmethod
    def get_steps(self, time: pd.Timestamp) -> Iterable:
        """Get available forecast steps or 'fxx' for this model run timestamp."""
        pass

    def get_data_vars(self, search: str) -> Iterable[str]:
        """
        Get available data_vars for the schema.
        """
        from herbie import Herbie

        H = Herbie("2023-01-01", model=self.name, fxx=0)
        data_vars = [
            name
            for name in H.inventory(search).variable.unique()
            # funny unknown HRRR variable
            if not name.startswith("var discipline=")
        ]
        return data_vars

    def get_available_times(self, since, till=None):
        if till is None:
            till = datetime.utcnow()
        since = pd.Timestamp(since).floor(self.update_freq)
        till = pd.Timestamp(till).floor(self.update_freq)
        available_times = pd.date_range(since, till, inclusive="left", freq=self.update_freq)
        if available_times.empty:
            raise RuntimeError(f"No data available for time range {since!r} to {till!r}")
        return available_times

    def latest(self):
        from herbie import HerbieLatest

        return HerbieLatest(model=self.name)

    def open_herbie(self, job: Job) -> xr.Dataset:
        from herbie import FastHerbie

        FH = FastHerbie(
            DATES=[job.runtime],
            fxx=list(job.steps),
            model=self.name,
            product=job.ingest.product,
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


def get_repo(name):
    import arraylake as al

    client = al.Client()
    return client.get_or_create_repo(f"earthmover-demos/{name}")


def create_repo(name: str):
    import arraylake as al

    client = al.Client()
    with contextlib.suppress(ValueError):
        client.delete_repo(f"earthmover-demos/{name}", imsure=True, imreallysure=True)

    return client.create_repo(f"earthmover-demos/{name}")


def optimize_coord_encoding(values, dx, is_regular=False):
    if is_regular:
        dx_all = np.diff(values)
        np.testing.assert_allclose(dx_all, dx), "must be regularly spaced"

    offset_codec = zarr.FixedScaleOffset(
        offset=values[0], scale=1 / dx, dtype=values.dtype, astype="i8"
    )
    delta_codec = zarr.Delta("i8", "i2")
    compressor = zarr.Blosc(cname="zstd")

    enc0 = offset_codec.encode(values)
    if is_regular:
        # everything should be offset by 1 at this point
        np.testing.assert_equal(np.unique(np.diff(enc0)), [1])
    enc1 = delta_codec.encode(enc0)
    # now we should be able to compress the shit out of this
    enc2 = compressor.encode(enc1)
    decoded = offset_codec.decode(delta_codec.decode(compressor.decode(enc2)))

    # will produce numerical precision differences
    np.testing.assert_equal(values, decoded)
    # np.testing.assert_allclose(values, decoded)

    return {"compressor": compressor, "filters": (offset_codec, delta_codec)}


def create_time_encoding(freq: timedelta) -> dict:
    """
    Creates a time encoding.
    """
    from xarray.conventions import encode_cf_variable

    time = xr.Variable(
        data=pd.date_range("2000-04-01", "2035-05-01", freq=freq),
        dims=("time",),
    )
    encoded = encode_cf_variable(time)
    time_values = encoded.data
    compression = optimize_coord_encoding(time_values, dx=freq.seconds / 3600, is_regular=True)

    encoding = encoded.encoding
    encoding.update(compression)
    encoding["chunks"] = (120,)

    return encoding
