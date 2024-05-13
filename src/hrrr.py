from datetime import datetime, timedelta
from typing import Iterable

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from . import lib
from .lib import RENAME_VARS, ForecastModel

logger = lib.get_logger()


class HRRR(ForecastModel):
    name = "hrrr"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    dim_order = ("x", "y", "time", "step")
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
        # add one for the "analysis"
        if (time - time.floor("D")) % timedelta(hours=6) == timedelta(hours=0):
            return range(49)
        else:
            return range(19)

    def create_schema(self, search: str, times=None) -> xr.Dataset:
        """
        Create schema Xarray Dataset for a list of model run times.
        """
        if times is None:
            times = [datetime.utcnow()]

        schema = xr.Dataset()

        schema["time"] = ("time", times)
        schema["time"].encoding.update(lib.create_time_encoding(self.update_freq))
        schema["time"].encoding["standard_name"] = "forecast_reference_time"

        schema["step"] = ("step", pd.to_timedelta(np.arange(49), unit="hours"))
        schema["step"].encoding.update(
            lib.optimize_coord_encoding(
                (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
            )
        )
        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"
        schema["step"].encoding["standard_name"] = "forecast_period"

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

        shape = tuple(schema.sizes[dim] for dim in self.dim_order)
        # TODO: Make this configurable
        chunksizes = {"x": 360, "y": 120, "time": 1, "step": 19}
        chunks = tuple(chunksizes[dim] for dim in self.dim_order)
        for name in self.get_data_vars(search):
            name = RENAME_VARS.get(name, name).lower()
            schema[name] = (self.dim_order, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
            schema[name].encoding["chunks"] = chunks
            schema[name].encoding["write_empty_chunks"] = False
        return schema
