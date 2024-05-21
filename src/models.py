from .gfs import GFS
from .hrrr import HRRR
from .lib import ForecastModel


def get_model(name: str) -> ForecastModel:
    match name.lower():
        case "gfs":
            return GFS()
        case "hrrr":
            return HRRR()
        case _:
            raise ValueError
