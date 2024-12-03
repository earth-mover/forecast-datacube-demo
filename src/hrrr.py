from datetime import datetime, timedelta
from typing import Sequence

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from . import lib
from .lib import ForecastModel, Ingest, merge_searches

logger = lib.get_logger()


VERTICAL_COORD_ATTRS = {
    "isobaricInhPa": {
        "long_name": "pressure",
        "units": "hPa",
        "positive": "down",
        "stored_direction": "decreasing",
        "standard_name": "air_pressure",
    },
}


class HRRR(ForecastModel):
    name = "hrrr"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    dim_order = ("isobaricInhPa", "x", "y", "time", "step")
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
        return lat.astype("float32"), lon.astype("float32")

    def get_steps(self, time: pd.Timestamp) -> Sequence:
        # 48 hour forecasts every 6 hours, 18 hour forecasts otherwise
        # add one for the "analysis"
        if (time - time.floor("D")) % timedelta(hours=6) == timedelta(hours=0):
            return range(49)
        else:
            return range(19)

    def create_schema(self, ingest: Ingest, *, times=None) -> xr.Dataset:
        """
        Create schema Xarray Dataset for a list of model run times.
        """
        chunksizes = ingest.chunks
        renames = ingest.renames
        search = merge_searches(ingest.searches)

        if times is None:
            times = [datetime.utcnow()]

        schema = xr.Dataset()

        schema["time"] = ("time", times, {"standard_name": "forecast_reference_time"})
        schema["time"].encoding.update(lib.create_time_encoding(self.update_freq))

        if search is None:
            schema["step"] = ("step", pd.to_timedelta(np.arange(49), unit="hours"))
            schema["step"].encoding.update(
                lib.optimize_coord_encoding(
                    (schema.step.data / 1e9 / 3600).astype(int), dx=1, is_regular=False
                )
            )
        else:
            schema["step"] = (
                "step",
                pd.to_timedelta(self.get_steps_for_search(search), unit="hour"),
            )
        schema["step"].encoding["chunks"] = schema.step.shape
        schema["step"].encoding["units"] = "hours"
        schema["step"].attrs["standard_name"] = "forecast_period"

        if search is not None:
            coord, levels = self.get_levels_for_search(search, product=ingest.product)
            if levels:
                assert coord == "isobaricInhPa"
                schema[coord] = (
                    coord,
                    np.array(levels, dtype=np.int16),
                    VERTICAL_COORD_ATTRS[coord],
                )
                schema[coord].encoding["chunks"] = len(levels)
                schema["step"].encoding.update(
                    lib.optimize_coord_encoding(schema[coord].data, dx=25, is_regular=True)
                )

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
        schema.latitude.encoding["chunks"] = schema.latitude.shape
        schema.longitude.encoding["chunks"] = schema.longitude.shape

        schema.coords["spatial_ref"] = (
            tuple(),
            0,
            {
                "crs_wkt": "".join(
                    [
                        'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unk',
                        'nown",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenw',
                        'ich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVER',
                        'SION["unknown",METHOD["Lambert Conic Conformal',
                        '(2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",38.5,ANGL',
                        'EUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitu',
                        'de of false origin",262.5,ANGLEUNIT["degree",0.0174532925199433],ID["E',
                        'PSG",8822]],PARAMETER["Latitude of 1st standard parallel",38.5,ANGLEUN',
                        'IT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude',
                        'of 2nd standard parallel",38.5,ANGLEUNIT["degree",0.0174532925199433],',
                        'ID["EPSG",8824]],PARAMETER["Easting at false',
                        'origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing',
                        'at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesia',
                        'n,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],A',
                        'XIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]',
                    ]
                ),
                "semi_major_axis": 6371229.0,
                "semi_minor_axis": 6371229.0,
                "inverse_flattening": 0.0,
                "longitude_of_prime_meridian": 0.0,
                "prime_meridian_name": "Greenwich",
                # 'reference_ellipsoid_name': 'unknown',
                # 'geographic_crs_name': 'unknown',
                # 'horizontal_datum_name': 'unknown',
                # 'projected_crs_name': 'unknown',
                "grid_mapping_name": "lambert_conformal_conic",
                "standard_parallel": (38.5, 38.5),
                "latitude_of_projection_origin": 38.5,
                "longitude_of_central_meridian": 262.5,
                "false_easting": 0.0,
                "false_northing": 0.0,
                "long_name": "HRRR model grid projection",
            },
        )

        schema.attrs = {
            "coordinates": "latitude longitude spatial_ref",
            "description": "HRRR data ingested for forecasting demo",
        }

        if search is None:
            return schema

        # TODO: refactor to helper func
        dim_order = tuple(dim for dim in self.dim_order if dim in schema.dims)
        shape = tuple(schema.sizes[dim] for dim in dim_order)
        chunks = tuple(chunksizes[dim] for dim in dim_order)
        for name in self.get_data_vars(search, renames=renames):
            schema[name] = (dim_order, dask.array.ones(shape, chunks=chunks, dtype=np.float32))
            schema[name].encoding["chunks"] = chunks
            schema[name].encoding["write_empty_chunks"] = False
        return schema
