import time
from datetime import datetime, timedelta
from typing import Hashable, Sequence

import dask.array
import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from . import lib
from .lib import ForecastModel, open_single_grib

logger = lib.get_logger()


class GFS(ForecastModel):
    # Product specific kwargs
    name = "gfs"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    update_freq = timedelta(hours=6)
    dim_order = ("longitude", "latitude", "time", "step")

    def get_steps(self, time: pd.Timestamp) -> Sequence:
        return list(range(0, 120)) + list(range(120, 385, 3))

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
        expand_dims: Sequence[Hashable] | None = None,
        drop_vars: Sequence[Hashable] | None = None,
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

    def create_schema(
        self,
        chunksizes: dict[str, int],
        *,
        renames: dict[str, str] | None,
        search: str | None = None,
        times=None,
    ) -> xr.Dataset:
        """
        Create schema Xarray Dataset for a list of model run times.
        """
        if times is None:
            times = [datetime.utcnow()]
        schema = xr.Dataset()
        schema["latitude"] = (
            "latitude",
            np.arange(90, -90.1, -0.25),
            {"standard_name": "latitude", "units": "degrees_north"},
        )
        schema["longitude"] = (
            "longitude",
            np.arange(0, 360, 0.25),
            {"standard_name": "longitude", "units": "degrees_east"},
        )
        schema["time"] = ("time", times, {"standard_name": "forecast_reference_time"})
        if search is not None:
            schema["step"] = (
                "step",
                pd.to_timedelta(self.get_steps(pd.Timestamp(datetime.utcnow())), unit="hours"),
            )
            schema["step"].encoding.update(
                lib.optimize_coord_encoding(
                    (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
                )
            )
        else:
            schema["step"] = (
                "step",
                pd.to_timedelta(self.get_steps_for_search(search), unit="hours"),  # type: ignore
            )

        schema["longitude"].encoding.update(
            lib.optimize_coord_encoding(schema["latitude"].data, dx=-0.25, is_regular=True)
        )
        schema["longitude"].encoding["chunks"] = schema.longitude.shape

        schema["latitude"].encoding.update(
            lib.optimize_coord_encoding(schema["longitude"].data, dx=0.25, is_regular=True)
        )
        schema["latitude"].encoding["chunks"] = schema.latitude.shape

        schema["time"].encoding.update(lib.create_time_encoding(self.update_freq))

        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"
        schema["step"].attrs["standard_name"] = "forecast_period"

        if search is None:
            return schema

        shape = tuple(schema.sizes[dim] for dim in self.dim_order)
        chunks = tuple(chunksizes[dim] for dim in self.dim_order)
        for name in self.get_data_vars(search=search, renames=renames):
            schema[name] = (self.dim_order, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
            schema[name].encoding["chunks"] = chunks
        return schema
