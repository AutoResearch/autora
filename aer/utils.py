from datetime import datetime
from enum import Enum


class Plot_Types(Enum):
    LINE = 1
    IMAGE = 2
    LINE_SCATTER = 3
    SURFACE_SCATTER = 4
    MULTI_LINE = 5
    MODEL = 6


def print_current_time():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    return


def do_nothing(*args, **kwargs):
    pass
