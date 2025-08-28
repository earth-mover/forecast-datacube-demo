#!/usr/bin/env python3

from datetime import datetime, timedelta

import modal

from src.lib import ReadMode, WriteMode
from src.lib_modal import MODAL_FUNCTION_KWARGS
from src.modal_app import applib, driver

app = modal.App("arraylake-hrrr")
app.include(applib)  # necessary


@app.local_entrypoint()
def main(mode: str, toml_file: str, since: str, till: str | None = None):
    if mode != "backfill":
        raise ValueError("Only mode='backfill' is allowed.")

    driver(mode=WriteMode.BACKFILL, toml_file_path=toml_file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, schedule=modal.Cron("*/30 * * * *"), timeout=300)
def hrrr_verify():
    driver(mode=ReadMode.VERIFY, toml_file_path="src/configs/hrrr-integration.toml")


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600 * 3)
def hrrr_backfill():
    """Run this "backfill" function wtih `modal run modal_hrrr.py::hrrr_backfill`."""
    file = "src/configs/hrrr-integration.toml"
    mode = WriteMode.BACKFILL
    since = datetime(2025, 8, 12)
    till = datetime.now() - timedelta(days=1, hours=0)
    # till = datetime(2025, 2, 4, 0, 0, 0)

    driver(mode=mode, toml_file_path=file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, timeout=20 * 60, schedule=modal.Cron("57 * * * *"))
def hrrr_update_solar():
    """Run this "backfill" function wtih `modal run modal_hrrr.py::hrrr_backfill`."""
    file = "src/configs/hrrr-integration.toml"
    mode = WriteMode.UPDATE
    driver(mode=mode, toml_file_path=file)
