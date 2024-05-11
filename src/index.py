import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexes import Index, PandasIndex
from xarray.core.indexing import IndexSelResult

Timestamp = str | datetime.datetime | pd.Timestamp | np.datetime64
Timedelta = str | datetime.timedelta | np.timedelta64  # TODO: pd.DateOffset also?


####################
####  Indexer types
# reference: https://www.unidata.ucar.edu/presentations/caron/FmrcPoster.pdf


@dataclass(init=False)
class ModelRun:
    """
    The complete results for a single run is a model run dataset.

    Parameters
    ----------
    time: Timestamp-like
        Initialization time for model run
    """

    time: pd.Timestamp

    def __init__(self, time: Timestamp):
        self.time = pd.Timestamp(time)


@dataclass(init=False)
class ConstantOffset:
    """
    A constant offset dataset is created from all the data that have the same offset time.
    Offset here refers to a variable usually named `step` or `lead` or with CF standard name
    `forecast_period`.
    """

    step: pd.Timedelta

    def __post_init__(self, step: Timedelta):
        self.step = pd.Timedelta(step)


@dataclass(init=False)
class ConstantForecast:
    """
    A constant forecast dataset is created from all the data that have the same forecast/valid time.

    Parameters
    ----------
    time: Timestamp-like

    """

    time: pd.Timestamp

    def __init__(self, time: Timestamp):
        self.time = pd.Timestamp(time)

    def get_indexer(self, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex) -> dict:
        target = self.time
        max_timedelta = period_index[-1]

        # earliest timestep we can start at
        earliest = target - max_timedelta
        left = time_index.get_slice_bound(earliest, side="left")

        # latest we can get
        right = time_index.get_slice_bound(target, side="right")
        # print(left, right)

        needed_times = time_index[slice(left, right)]
        needed_steps = target - needed_times
        # print(needed_times, needed_steps)

        needed_time_idxs = np.arange(left, right)
        needed_step_idxs = period_index.get_indexer(needed_steps)
        # print(needed_time_idxs, needed_step_idxs)

        # It's possible we don't have the right step.
        # If pandas doesn't find an exact match it returns -1.
        mask = needed_step_idxs != -1

        needed_step_idxs = needed_step_idxs[mask]
        needed_time_idxs = needed_time_idxs[mask]

        assert needed_step_idxs.size == needed_time_idxs.size

        return needed_time_idxs, needed_step_idxs


@dataclass
class BestEstimate:
    """
    For each forecast time in the collection, the best estimate for that hour is used to create a
    best estimate dataset, which covers the entire time range of the collection.
    """

    # TODO: `since` could be a timedelta relative to `asof`.
    # TODO: could have slice(since, asof)
    since: pd.Timestamp | None = None
    asof: pd.Timestamp | None = None

    def __post_init__(self):
        if self.asof is not None and self.since is not None and self.asof < self.since:
            raise ValueError(
                "Can't request best estimate since {since=!r} "
                "which is earlier than requested {asof=!r}"
            )

    def get_indexer(self, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex):
        if period_index[0] != pd.Timedelta(0):
            raise ValueError(
                "Can't make a best estimate dataset if forecast_period doesn't start at 0."
            )

        # TODO: consolidate the get_indexer lookup
        if self.since is None:
            first_index = 0
        else:
            (first_index,) = time_index.get_indexer([self.since])

        if self.asof is None:
            last_index = time_index.size - 1
        else:
            (last_index,) = time_index.get_indexer([self.asof])

        needed_time_idxrs = np.concatenate(
            [
                np.arange(first_index, last_index, dtype=int),
                np.repeat(last_index, period_index.size),
            ]
        )
        needed_step_idxrs = np.concatenate(
            [np.zeros((last_index - first_index,), dtype=int), np.arange(period_index.size)]
        )

        return needed_time_idxrs, needed_step_idxrs


@dataclass
class Indexes:
    reference_time: PandasIndex
    period: PandasIndex
    # valid_time: xr.Variable


class ForecastIndex(Index):
    # based off Benoit's RasterIndex in
    # https://hackmd.io/Zxw_zCa7Rbynx_iJu6Y3LA?view

    def __init__(self, variables: Indexes, dummy_name: str):
        self._indexes = variables

        assert isinstance(dummy_name, str)
        self.dummy_name = dummy_name

        self.names = {
            "reference_time": self._indexes.reference_time.index.name,
            "period": self._indexes.period.index.name,
        }

    @classmethod
    def from_variables(cls, variables, options):
        """
        Must be created from three variables:
        1. A dummy scalar `forecast` variable.
        2. A variable with the CF attribute`standard_name: "forecast_reference_time"`.
        3. A variable with the CF attribute`standard_name: "forecast_period"`.
        """
        assert len(variables) == 3

        dummy_name = None

        indexes = {}
        for k in ["forecast_reference_time", "forecast_period"]:
            for name, var in variables.items():
                std_name = var.attrs.get("standard_name", None)
                if k == std_name:
                    indexes[k.removeprefix("forecast_")] = PandasIndex.from_variables(
                        {name: var}, options=None
                    )
                elif var.ndim == 0:
                    dummy_name = name

        return cls(Indexes(**indexes), dummy_name=dummy_name, **options)

    def sel(self, labels, **kwargs):
        """
        Allows three kinds of indexing
        1. Along the dummy "forecast" variable: enable specialized methods using
           ConstantOffset, ModelRun, ConstantForecast, BestEstimate
        2. Along the `forecast_reference_time` dimension, identical to ModelRun
        3. Along the `forecast_period` dimension, indentical to ConstantOffset

        You cannot mix (1) with (2) or (3), but (2) and (3) can be combined in a single
        statement.
        """
        if self.dummy_name in labels:
            assert len(labels) == 1

            label: ConstantOffset | ModelRun | ConstantForecast | BestEstimate
            label = next(iter(labels.values()))

            time_index = self._indexes.reference_time.index
            period_index = self._indexes.period.index

            match label:
                case ConstantOffset(step):
                    indexer = {self.names["period"]: period_index.get_indexer(step)}

                case ModelRun(timestamp):
                    indexer = {self.names["reference_time"]: time_index.get_indexer(timestamp)}

                case ConstantForecast() | BestEstimate():
                    time_idxrs, period_idxrs = label.get_indexer(time_index, period_index)
                    indexer = {
                        self.names["reference_time"]: xr.Variable("valid_time", time_idxrs),
                        self.names["period"]: xr.Variable("valid_time", period_idxrs),
                    }

                case _:
                    raise ValueError(f"Invalid indexer type {type(label)} for label: {label}")

            match label:
                case ConstantForecast():
                    new_indexers = {"valid_time": xr.Variable("valid_time", [label.time])}
                case BestEstimate():
                    new_indexers = {
                        "valid_time": time_index[time_idxrs] + period_index[period_idxrs]
                    }
                case _:
                    new_indexers = {}

        # sel needs to only handle keys in labels
        # since it delegates to isel.
        # we handle all entries in ._indexes there
        return IndexSelResult(
            dim_indexers=indexer, variables=new_indexers, drop_coords=["forecast"]
        )

    def __repr__(self):
        string = (
            f"<ForecastIndex along [{', '.join([self.dummy_name] + list(self.names.values()))}]>"
        )
        return string
