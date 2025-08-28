from collections.abc import Sequence
from datetime import timedelta

import dask.array
import numpy as np
import pandas as pd
import pyproj
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


HRRR_CRS_WKT = "".join(
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
)


class HRRR(ForecastModel):
    name = "hrrr"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    dim_order = ("isobaricInhPa", "x", "y", "time", "step")
    update_freq = timedelta(hours=1)

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
    # https://github.com/blaylockbk/Herbie/discussions/45#discussioncomment-8570650
    transformer = pyproj.Transformer.from_crs(HRRR_CRS_WKT, 4326, always_xy=True)
    lon0 = 237.280472
    lat0 = 21.138123
    dx = dy = 3000

    def get_lat_lon(self):
        """Generate lat, lon for HRRR grid. Used for schema."""
        dx, dy = self.dx, self.dy
        Nx, Ny = 1799, 1059
        x0, y0 = self.transformer.transform(self.lon0, self.lat0, direction="INVERSE")
        x, y = np.meshgrid(np.arange(x0, x0 + dx * Nx, dx), np.arange(y0, y0 + dy * Ny, dy))
        lon, lat = self.transformer.transform(x, y)
        lon += 360
        return lat.astype("float32"), lon.astype("float32")

    def get_geotransform(self):
        dx, dy = self.dx, self.dy
        x0, y0 = self.transformer.transform(self.lon0, self.lat0, direction="INVERSE")
        return f"{x0-dx/2} {dx} 0 {y0-dy/2} 0 {dy}"

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
            times = [lib.utcnow()]

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
        schema["step"].encoding["dtype"] = "timedelta64[h]"
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
                "crs_wkt": HRRR_CRS_WKT,
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
                "GeoTransform": self.get_geotransform(),
            },
        )

        schema.coords["crs_4326"] = (
            tuple(),
            0,
            {
                "crs_wkt": 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.314245179,
                "inverse_flattening": 298.257223563,
                "reference_ellipsoid_name": "WGS 84",
                "longitude_of_prime_meridian": 0.0,
                "prime_meridian_name": "Greenwich",
                "geographic_crs_name": "WGS 84",
                "horizontal_datum_name": "World Geodetic System 1984 ensemble",
                "grid_mapping_name": "latitude_longitude",
            },
        )

        schema.attrs = {
            "coordinates": "latitude longitude spatial_ref crs_4326",
            "description": "HRRR data ingested for forecasting demo",
        }

        if search is None:
            return schema

        # TODO: refactor to helper func
        dim_order = tuple(dim for dim in self.dim_order if dim in schema.dims)
        shape = tuple(schema.sizes[dim] for dim in dim_order)
        chunks = tuple(chunksizes[dim] for dim in dim_order)
        for name in self.get_data_vars(search, renames=renames):
            schema[name] = (
                dim_order,
                dask.array.ones(shape, chunks=(-1,) * len(chunks), dtype=np.float32),
            )
            schema[name].encoding["chunks"] = chunks
            schema[name].encoding["write_empty_chunks"] = False
            schema[name].attrs["grid_mapping"] = "spatial_ref: crs_4326: latitude longitude"
        return schema
