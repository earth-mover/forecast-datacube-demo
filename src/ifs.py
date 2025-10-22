from collections.abc import Hashable, Sequence
from datetime import timedelta

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from . import lib
from .lib import ForecastModel, IndexColumns, Ingest, merge_searches

logger = lib.get_logger()


class IFS(ForecastModel):
    # Product specific kwargs
    name = "ifs"
    runtime_dim = "time"
    step_dim = "step"
    expand_dims = ("step", "time")
    drop_vars = ("valid_time",)
    update_freq = timedelta(hours=12)
    dim_order = ("longitude", "latitude", "time", "step")

    columns = IndexColumns(
        level="levelist",
        variable="param",
        valid_time="valid_time",
    )

    def get_steps(self, time: pd.Timestamp) -> Sequence:
        return np.arange(0, 144, 3).tolist() + np.arange(144, 241, 6).tolist()

    def get_urls(self, time: pd.Timestamp) -> list[str]:
        """
        Returns list of urls given a model run timestamp.
        """
        raise NotImplementedError

    def open_single_grib(
        self,
        uri,
        expand_dims: Sequence[Hashable] | None = None,
        drop_vars: Sequence[Hashable] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        raise NotImplementedError

    def open_multiple_gribs(self, urls, expand_dims=None, drop_vars=None, **kwargs):
        raise NotImplementedError

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
        schema["latitude"] = (
            "latitude",
            np.arange(90, -90.1, -0.25),
            {"standard_name": "latitude", "units": "degrees_north"},
        )
        schema["longitude"] = (
            "longitude",
            np.arange(-180, 180, 0.25),
            {"standard_name": "longitude", "units": "degrees_east"},
        )
        schema["time"] = ("time", times, {"standard_name": "forecast_reference_time"})
        if search is not None:
            schema["step"] = (
                "step",
                pd.to_timedelta(self.get_steps(pd.Timestamp(lib.utcnow())), unit="hours"),
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
        schema["step"].encoding["dtype"] = "timedelta64[h]"
        schema["step"].attrs["standard_name"] = "forecast_period"

        schema.attrs = {
            "description": "IFS data ingested for forecasting demo",
        }

        if search is None:
            return schema

        # TODO: refactor to helper func
        dim_order = tuple(dim for dim in self.dim_order if dim in schema.dims)
        shape = tuple(schema.sizes[dim] for dim in dim_order)
        chunks = tuple(chunksizes[dim] for dim in dim_order)
        for name in self.get_data_vars(search=search, renames=renames):
            schema[name] = (
                dim_order,
                dask.array.ones(shape, chunks=(-1,) * len(chunks), dtype=np.float32),
            )
            schema[name].encoding["chunks"] = chunks
        return schema
