import contextlib
import itertools
import logging
import random
import string
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from typing import Any, Literal

import cfgrib
import fsspec
import numcodecs
import numpy as np
import pandas as pd
import tomllib
import xarray as xr

TimestampLike = Any


def utcnow():
    # herbie requires timezone-naive timestamps
    return datetime.now(UTC).replace(tzinfo=None)


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
    renames: dict[str, str] | None = None
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

    # Important: `time` and `step` (or `fxx`) can be passed to Herbie directly
    # to filter results.
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
    def create_schema(self, ingest: Ingest, *, times=None) -> xr.Dataset:
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "This pattern is interpreted as a regular expression, and has match groups. "
                    "To actually get the groups, use str.extract."
                ),
                category=UserWarning,
            )

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "This pattern is interpreted as a regular expression, and has match groups. "
                    " To actually get the groups, use str.extract."
                ),
                category=UserWarning,
            )

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "This pattern is interpreted as a regular expression, and has match groups. "
                    " To actually get the groups, use str.extract."
                ),
                category=UserWarning,
            )

            unique_steps = H.inventory(search).forecast_time.unique()
        return [0 if s == "anl" else int(s.removesuffix(" hour fcst")) for s in unique_steps]

    def get_levels_for_search(self, search: str, *, product: str) -> tuple[str, list[int]]:
        """
        These are `fxx` or `step` values available for this particular search.
        We have to execute the `search` in case the search string constrains the step values.
        """
        from herbie import FastHerbie

        time = pd.Timestamp("2023-01-01")
        H = FastHerbie([time], model=self.name, product=product, fxx=self.get_steps(time))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "This pattern is interpreted as a regular expression, and has match groups. "
                    " To actually get the groups, use str.extract."
                ),
                category=UserWarning,
            )

            unique_levels = H.inventory(search).level.unique()
        # TODO: really need a better way to handle vertical levels
        if len(unique_levels) > 1 and all(" mb" in level for level in unique_levels):
            logger.debug(f"Returning isobaricInhPa for {unique_levels=!r}")
            return "isobaricInhPa", [int(s.removesuffix(" mb")) for s in unique_levels]
        else:
            return unique_levels[0], []

    def get_available_times(
        self, since: TimestampLike, till: TimestampLike | None = None
    ) -> pd.DatetimeIndex:
        """
        These are expected timestamps that are available for this particular model.
        """
        if till is None:
            till = utcnow()
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
        """Opens the GRIB files specified in job.ingest and returns a Dataset."""
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

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "This pattern is interpreted as a regular expression, and has match groups. "
                    " To actually get the groups, use str.extract."
                ),
                category=UserWarning,
            )
            inv = FH.inventory(search=search)
            if inv.forecast_time.nunique() != len(job.steps):
                raise ValueError(f"Not all files are available for job: {job!r}")

            paths = FH.download(search=search)
            logger.debug("Downloaded paths {}".format(paths))

        # TODO: really need a better way to handle vertical levels
        unique_levels = inv.level.unique()
        if len(unique_levels) > 1 and all(" mb" in level for level in unique_levels):
            logger.debug(f"Returning isobaricInhPa for {unique_levels=!r}")
            level_dim, levels = "isobaricInhPa", [int(s.removesuffix(" mb")) for s in unique_levels]
        else:
            level_dim, levels = None, []
        ds = xr.combine_nested(
            [
                xr.merge(cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}))
                for path in sorted(paths)
            ],
            concat_dim="step",
        )
        ds = ds.expand_dims("time").sortby("step")
        if levels:
            ds = ds.reindex({level_dim: levels})

        counts = ds.count("step").compute()
        if not levels and not (counts == len(job.steps)).to_array().all().item():
            # 3D datasets have lots of corrupt data!
            raise ValueError(f"This dataset has NaNs. Aborting \n{counts}")

        # TODO: could be more precise here.
        dim_order = tuple(dim for dim in self.dim_order if dim in ds.dims)
        return ds.chunk(step=job.ingest.chunks["step"]).transpose(*dim_order)


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


def maybe_get_repo(name, client=None):
    import arraylake as al

    if client is None:
        client = al.Client()

    ALPREFIX = "arraylake://"
    ICEPREFIX = "icechunk://"
    if name.startswith(ALPREFIX):
        logger.info(f"Opening Arraylake store: {name!r}")
        return client.get_or_create_repo(name.removeprefix(ALPREFIX))
    elif name.startswith(ICEPREFIX):
        logger.info(f"Opening Icechunk store: {name!r}")
        return client.get_or_create_repo(name.removeprefix(ICEPREFIX), kind=al.types.RepoKind.V2)
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

    offset_codec = numcodecs.FixedScaleOffset(
        offset=values[0], scale=1 / dx, dtype=values.dtype, astype="i8"
    )
    delta_codec = numcodecs.Delta("i8", "i2")
    compressor = numcodecs.Blosc(cname="zstd")

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


def parse_toml_config(file: str) -> dict[str, Ingest]:
    from . import models

    with open(file, mode="rb") as f:
        parsed = tomllib.load(f)

    ingest_jobs = {}
    for key, values in parsed.items():
        model = models.get_model(values["model"])
        if unknown_dims := (set(values["chunks"]) - set(model.dim_order)):
            raise ValueError(
                f"Unrecognized dimension names in chunks: {unknown_dims}. "
                f"Expected {model.dim_order!r}."
            )
        ingest_jobs[key] = Ingest(name=key, **values)
    return ingest_jobs


def merge_searches(searches: Sequence[str]) -> str:
    """
    Merges a string of `searches` together.
    Assuming that a simple `|` will do sensible things.
    """
    return "|".join(searches)
