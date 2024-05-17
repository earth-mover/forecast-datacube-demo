import contextlib
import itertools
import logging
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum, auto
from typing import Any, Hashable, Iterable, Literal, Sequence

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr

TimestampLike = Any


def get_logger():
    logger = logging.getLogger("modal_app")
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def random_string(n):
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


class WriteMode(StrEnum):
    BACKFILL = auto()
    UPDATE = auto()


@dataclass
class Ingest:
    """
    Defines the ingestion of a bunch of GRIB variables selected
    by the Herbie search string ``search``, from the output of
    ``model`` and ``product``, and written to a Zarr group ``zarr_group``
    in the Zarr store ``store``.
    """

    name: str
    model: Literal["gfs", "hrrr"]
    product: str
    store: str
    zarr_group: str
    searches: Sequence[str]
    chunks: dict[str, int]
    renames: list[str] | None = None
    zarr_store: Any = None

    def __iter__(self):
        for search in self.searches:
            yield type(self)(
                searches=[search],
                name=self.name,
                model=self.model,
                product=self.product,
                store=self.store,
                zarr_group=self.zarr_group,
                chunks=self.chunks,
                renames=self.renames,
                zarr_store=self.zarr_store,
            )


@dataclass
class Job:
    """
    Defines an ``Ingest`` for the model start time ``runtime`` for the forecast
    time steps ``steps``.
    """

    runtime: pd.Timestamp
    steps: list[int]
    ingest: Ingest


class ForecastModel(ABC):
    name: str
    runtime_dim: str
    step_dim: str
    expand_dims: Sequence[str]
    drop_vars: Sequence[str]
    dim_order: Sequence[str]
    update_freq: timedelta

    @abstractmethod
    def create_schema(
        self,
        chunksizes: dict[str, int],
        *,
        renames: dict[str, str] | None,
        search: str | None = None,
        times=None,
    ) -> xr.Dataset:
        """Create schema with chunking for on-disk storage."""
        pass

    @abstractmethod
    def get_steps(self, time: pd.Timestamp) -> Sequence:
        """Get available forecast steps or 'fxx' for this model run timestamp."""
        pass

    def as_xarray(self, search: str) -> xr.Dataset:
        """
        Creates an Xarray dataset for a search. Necessary for populating attributes
        in the schema.
        """
        from herbie import Herbie

        H = Herbie("2023-01-01", model=self.name, fxx=0)
        dset = H.xarray(search)
        if isinstance(dset, list):
            dset = xr.merge(dset)
        return dset

    def get_data_vars(self, search: str, renames: dict[str, str] | None = None) -> Iterable[str]:
        """
        Get available data_vars for the schema by inspecting the inventory
        (or idx file).
        """
        from herbie import Herbie

        H = Herbie("2023-01-01", model=self.name, fxx=0)
        data_vars = [
            renames.get(name, name.lower()) if renames is not None else name
            for name in H.inventory(search).variable.unique()
            # funny unknown HRRR variable
            if not name.startswith("var discipline=")
        ]
        return data_vars

    def get_steps_for_search(self, search: str) -> list[int]:
        """
        These are `fxx` or `step` values available for this particular search.
        We have to execute the `search` in case the search string constrains the step values.
        """
        from herbie import FastHerbie

        time = pd.Timestamp("2023-01-01")
        H = FastHerbie([time], model=self.name, fxx=self.get_steps(time))
        unique_steps = H.inventory(search).forecast_time.unique()
        return [0 if s == "anl" else int(s.removesuffix(" hour fcst")) for s in unique_steps]

    def get_available_times(
        self, since: TimestampLike, till: TimestampLike | None = None
    ) -> pd.DatetimeIndex:
        """
        These are expected timestamps that are available for this particular model.
        """
        if till is None:
            till = datetime.utcnow()
        since = pd.Timestamp(since).floor(self.update_freq)  # type: ignore[arg-type]
        till = pd.Timestamp(till).floor(self.update_freq)  # type: ignore[arg-type]
        available_times = pd.date_range(since, till, inclusive="left", freq=self.update_freq)
        if available_times.empty:
            raise RuntimeError(f"No data available for time range {since!r} to {till!r}")
        return available_times

    def latest_available(self, ingest: Ingest) -> pd.Timestamp:
        """
        Find the latest data we can ingest.
        1. First use HerbieLatest to grab the latest available `.idx` file.
        2. Then repeat the search to make sure we have all expected time steps.
        3. If not, we assume the previous timestep is the latest complete dataset
           (Note we do not verify this by repeating the search.)

        This method is a bit wasteful, but makes the pipeline more robust.
        """
        from herbie import FastHerbie, HerbieLatest

        # TODO: Think about whether we need to execute the search to undersand availability
        # Right now we just see if all the `.idx` files exist for the steps we want.
        # Since one Ingest references one `product` at the moment, this seems OK.
        # If we allow multiple products in a single Ingest, then we might have to do something.
        # (search,) = ingest.searches
        # Get the latest idx
        HL = HerbieLatest(model=self.name, product=ingest.product)

        # May not be complete yet.
        steps = self.get_steps(HL.date)
        FH = FastHerbie(
            [HL.date], model=ingest.model, product=ingest.product, priority="aws", fxx=steps
        )
        if FH.file_not_exists:
            latest_available = HL.date - self.update_freq
            logger.info(
                f"Data not complete for date={HL.date!r}. "
                f"Processing till previous update {latest_available!r}"
            )
        else:
            latest_available = HL.date
        return latest_available

    def open_herbie(self, job: Job) -> xr.Dataset:
        from herbie import FastHerbie

        FH = FastHerbie(
            DATES=[job.runtime],
            fxx=list(job.steps),
            model=self.name,
            product=job.ingest.product,
            priority="aws",
        )
        (search,) = job.ingest.searches
        logger.debug("Searching %s", search)

        inv = FH.inventory(search=search)
        if inv.forecast_time.nunique() != len(job.steps):
            raise ValueError(f"Not all files are available for job: {job!r}")

        paths = FH.download(search=search)
        logger.debug("Downloaded paths {}".format(paths))

        ds = (
            xr.open_mfdataset(sorted(paths), combine="nested", concat_dim="step", engine="cfgrib")
            .expand_dims("time")
            .sortby("step")
        )

        counts = ds.count("step").compute()
        if not (counts == len(job.steps)).to_array().all().item():
            raise ValueError(f"This dataset has NaNs. Aborting \n{counts}")

        return ds.chunk(step=job.ingest.chunks["step"]).transpose(*self.dim_order)


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
    expand_dims: Sequence[Hashable] | None = None,
    drop_vars: Sequence[Hashable] | None = None,
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


def get_zarr_store(name):
    import arraylake as al

    ALPREFIX = "arraylake://"
    if name.startswith(ALPREFIX):
        client = al.Client()
        return client.get_or_create_repo(name.removeprefix(ALPREFIX)).store
    return name


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
