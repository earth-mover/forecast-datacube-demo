import time
from datetime import datetime, timedelta
from typing import Hashable, Iterable

import dask.array
import fsspec
import numpy as np
import pandas as pd
import xarray as xr

import lib
from lib import ForecastModel, open_single_grib

logger = lib.get_logger()


RENAME_VARS = {"UGRD": "u", "VGRD": "v"}

# logger = logging.getLogger("herbie")
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)


class HRRR(ForecastModel):
    name = "hrrr"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    dim_order = ("time", "step", "y", "x")
    update_freq = timedelta(hours=1)

    def get_lat_lon(self):
        """Generate lat, lon for HRRR grid. Used for schema."""
        # GRIB_gridType : lambert
        # GRIB_DxInMetres : 3000.0
        # GRIB_DyInMetres : 3000.0
        # GRIB_LaDInDegrees : 38.5
        # GRIB_Latin1InDegrees : 38.5
        # GRIB_Latin2InDegrees : 38.5
        # GRIB_LoVInDegrees : 262.5
        # GRIB_NV : 0
        # GRIB_Nx : 1799
        # GRIB_Ny : 1059
        # GRIB_gridDefinitionDescription :
        #     Lambert Conformal can be secant or tangent, conical or bipolar
        # GRIB_iScansNegatively : 0
        # GRIB_jPointsAreConsecutive : 0
        # GRIB_jScansPositively : 1
        # GRIB_latitudeOfFirstGridPointInDegrees : 21.138123
        # GRIB_longitudeOfFirstGridPointInDegrees : 237.280472

        import cartopy.crs as ccrs
        import pyproj

        # https://github.com/blaylockbk/Herbie/discussions/45#discussioncomment-8570650
        projection = ccrs.LambertConformal(
            central_longitude=262.5,  # GRIB_LoVInDegrees
            central_latitude=38.5,  # GRIB_LaDInDegrees : 38.5
            standard_parallels=(38.5, 38.5),  # (GRIB_Latin1InDegrees, GRIB_Latin2InDegrees)
            globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
        )
        transformer = pyproj.Transformer.from_crs(projection.to_wkt(), 4326, always_xy=True)

        dx = dy = 3000
        Nx, Ny = 1799, 1059
        x0, y0 = transformer.transform(237.280472, 21.138123, direction="INVERSE")
        x, y = np.meshgrid(np.arange(x0, x0 + dx * Nx, dx), np.arange(y0, y0 + dy * Ny, dy))
        lon, lat = transformer.transform(x, y)
        lon += 360
        return lat, lon

    def get_steps(self, time: pd.Timestamp) -> Iterable:
        # 48 hour forecasts every 6 hours, 18 hour forecasts otherwise
        if (time - time.floor("D")) % timedelta(hours=6) == timedelta(hours=0):
            return range(48)
        else:
            return range(18)

    def create_schema(self, search: str, times=None) -> xr.Dataset:
        """
        Create schema Xarray Dataset for a list of model run times.
        """
        if times is None:
            times = [datetime.utcnow()]

        schema = xr.Dataset()

        schema["time"] = ("time", times)
        schema["time"].encoding.update(lib.create_time_encoding(self.update_freq))

        schema["step"] = ("step", pd.to_timedelta(np.arange(48), unit="hours"))
        schema["step"].encoding.update(
            lib.optimize_coord_encoding(
                (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
            )
        )
        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"

        # TODO: optimize encoding for latitude, longitude
        lat, lon = self.get_lat_lon()
        schema.coords["longitude"] = (
            ("y", "x"),
            lon,
            {"standard_name": "longitude", "units": "degrees_east"},
        )
        schema.coords["latitude"] = (
            ("y", "x"),
            lat,
            {"standard_name": "latitude", "units": "degrees_north"},
        )

        dims = ("time", "step", "y", "x")
        shape = tuple(schema.sizes[dim] for dim in dims)
        # TODO: Make this configurable
        chunks = (1, 6, 120, 360)
        for name in self.get_data_vars(search):
            name = RENAME_VARS.get(name, name).lower()
            schema[name] = (dims, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
            schema[name].encoding["chunks"] = chunks
            schema[name].encoding["write_empty_chunks"] = False
        return schema


class GFS(ForecastModel):
    # Product specific kwargs
    name = "gfs"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    update_freq = timedelta(hours=6)
    dim_order = ("time", "step", "latitude", "longitude")

    def get_steps(self, time: pd.Timestamp) -> Iterable:
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

    def create_schema(self, search: str, times=None) -> xr.Dataset:
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
            lib.optimize_coord_encoding(schema["latitude"].data, dx=-0.25, is_regular=True)
        )
        schema["longitude"].encoding["chunks"] = schema.longitude.shape

        schema["latitude"].encoding.update(
            lib.optimize_coord_encoding(schema["longitude"].data, dx=0.25, is_regular=True)
        )
        schema["latitude"].encoding["chunks"] = schema.latitude.shape

        schema["time"].encoding.update(lib.create_time_encoding(self.update_freq))

        schema["step"].encoding.update(
            lib.optimize_coord_encoding(
                (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
            )
        )
        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"

        data_vars = self.get_data_vars()
        # TODO: Make this configurable
        dims = ("time", "step", "latitude", "longitude")
        shape = tuple(schema.sizes[dim] for dim in dims)
        chunks = (1, 24, 120, 360)
        for name in data_vars:
            schema[name] = (dims, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
            schema[name].encoding["chunks"] = chunks
        return schema
