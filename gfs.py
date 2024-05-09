import time
from datetime import datetime, timedelta
from typing import Hashable, Iterable

import dask.array
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr

import lib
from lib import ForecastModel, open_single_grib

logger = lib.get_logger()

# logger = logging.getLogger("herbie")
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)



class GFS(ForecastModel):
    # Product specific kwargs
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    update_freq = timedelta(hours=6)
    step = list(range(0, 120)) + list(range(120, 385, 3))

    def latest(self):
        from herbie import HerbieLatest

        return HerbieLatest(model="gfs")

    def get_urls(self, time: pd.Timestamp) -> list[str]:
        """
        Returns list of urls given a model run timestamp.
        """
        date_str = time.strftime("%Y%m%d")
        start = time.floor("6h").strftime("%H")
        print(date_str, start)
        fs = fsspec.filesystem("s3")
        urls = sorted(
            [
                "s3://" + f
                for f in fs.glob(
                    f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/{start}/atmos/gfs.t{start}z.pgrb2.0p25.f*"
                )
                if "idx" not in f
            ]
        )
        return urls

    def open_single_grib(
        self,
        uri,
        expand_dims: Iterable[Hashable] = None,
        drop_vars: Iterable[Hashable] = None,
        **kwargs,
    ) -> xr.Dataset:
        if expand_dims is None:
            expand_dims = self.expand_dims
        if drop_vars is None:
            drop_vars = self.drop_vars

        if uri.endswith(".f000"):
            # initialization is always stepType="instant".
            filter_by_keys = kwargs.pop("filter_by_keys", None)
            if filter_by_keys:
                filter_by_keys = {
                    k: "instant" if k == "stepType" and v != "instant" else v
                    for k, v in filter_by_keys.items()
                }
            kwargs["filter_by_keys"] = filter_by_keys
        tic = time.time()
        ds = open_single_grib(uri, expand_dims=expand_dims, drop_vars=drop_vars, **kwargs)
        logger.debug(f"Reading {uri} took {time.time() - tic} sec")

        return ds

    def open_multiple_gribs(self, urls, expand_dims=None, drop_vars=None, **kwargs):
        """Uses a threadpool to download the GRIB files, before opening them with xarray"""
        from concurrent.futures import ThreadPoolExecutor, wait

        def fsspec_open(uri: str) -> str:
            import time

            tic = time.time()
            # Make sure that fsspec uses the same names
            # so that we can infer whether we're opening the first grib.
            local_uri = fsspec.open_local(f"simplecache::{uri}", simplecache={"same_names": True})
            logger.debug(
                f"threadpool: Downloading uri {uri} to {local_uri} took {time.time() - tic} sec"
            )
            return local_uri

        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(fsspec_open, uri) for uri in urls]
            wait(futures)

        dsets = [
            self.open_single_grib(
                f.result(),
                expand_dims=expand_dims,
                drop_vars=drop_vars,
                **kwargs,
            )
            for f in futures
        ]
        return xr.concat(
            dsets,
            dim=self.step_dim,
            join="override",
            compat="override",
            data_vars="minimal",
            coords="minimal",
        )

    def create_schema(self, times=None):
        """
        Create schema Xarray Dataset for a list of model run times.
        """
        if times is None:
            times = [datetime.utcnow()]
        schema = xr.Dataset()
        schema["latitude"] = np.arange(90, -90.1, -0.25)
        schema["longitude"] = np.arange(0, 360, 0.25)
        schema["time"] = ("time", times)
        schema["step"] = ("step", pd.to_timedelta(self.step, unit="hours"))

        schema["longitude"].encoding.update(
            optimize_coord_encoding(schema["latitude"].data, dx=-0.25, is_regular=True)
        )
        schema["longitude"].encoding["chunks"] = schema.longitude.shape

        schema["latitude"].encoding.update(
            optimize_coord_encoding(schema["longitude"].data, dx=0.25, is_regular=True)
        )
        schema["latitude"].encoding["chunks"] = schema.latitude.shape

        schema["time"].encoding.update(create_time_encoding())

        schema["step"].encoding.update(
            optimize_coord_encoding(
                (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
            )
        )
        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"

        # TODO: Make this configurable
        dims = ("time", "step", "latitude", "longitude")
        shape = tuple(schema.sizes[dim] for dim in dims)
        chunks = (1, 24, 120, 360)
        schema["prate"] = (dims, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
        schema["prate"].encoding["chunks"] = chunks
        return schema

    def get_available_times(self, since, till=None):
        if till is None:
            till = datetime.utcnow()
        since = pd.Timestamp(since).floor(self.update_freq)
        till = pd.Timestamp(till).floor(self.update_freq)
        available_times = pd.date_range(since, till, inclusive="left", freq=self.update_freq)
        if available_times.empty:
            raise RuntimeError(f"No data available for time range {since!r} to {till!r}")
        return available_times


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


def create_time_encoding() -> dict:
    """
    Creates a time encoding.
    """
    from xarray.conventions import encode_cf_variable

    time = xr.Variable(
        data=pd.date_range("1970-04-01", "2035-05-01", freq="6h"),
        dims=("time",),
    )
    encoded = encode_cf_variable(time)
    time_values = encoded.data
    compression = optimize_coord_encoding(time_values, dx=6, is_regular=True)

    encoding = encoded.encoding
    encoding.update(compression)
    encoding["chunks"] = (120,)

    return encoding
