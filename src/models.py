from .gfs import GFS
from .hrrr import HRRR


def get_model(name: str):
    match name.lower():
        case "gfs":
            return GFS()
        case "hrrr":
            return HRRR()
        case _:
            raise ValueError
