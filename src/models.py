from .gfs import GFS
from .hrrr import HRRR
from .ifs import IFS
from .lib import ForecastModel


def get_model(name: str) -> ForecastModel:
    match name.lower():
        case "gfs":
            return GFS()
        case "hrrr":
            return HRRR()
        case "ifs":
            return IFS()
        case _:
            raise ValueError
