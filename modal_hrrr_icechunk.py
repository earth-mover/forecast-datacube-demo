from datetime import datetime, timedelta

import modal

from modal_app import applib, driver
from src.lib import WriteMode, utcnow
from src.lib_modal import MODAL_FUNCTION_KWARGS

app = modal.App("hrrr-icehunk-ingest")
app.include(applib)  # necessary


@app.local_entrypoint()
def main(mode: str, toml_file: str, since: str, till: str | None = None):
    if mode != "backfill":
        raise ValueError("Only mode='backfill' is allowed.")

    driver(mode=WriteMode.BACKFILL, toml_file_path=toml_file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, timeout=7200)
def hrrr_backfill_icechunk():
    """Run this "backfill" function wtih `modal run modal_hrrr.py::hrrr_backfill`."""
    file = "src/configs/hrrr-icechunk.toml"
    mode = WriteMode.BACKFILL
    since = datetime(2024, 11, 1)
    till = utcnow() - timedelta(days=1, hours=12)

    driver(mode=mode, toml_file_path=file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, timeout=1200, schedule=modal.Cron("57 * * * *"))
def hrrr_update_solar_icechunk():
    """Run this "backfill" function wtih `modal run modal_hrrr.py::hrrr_backfill`."""
    file = "src/configs/hrrr-icechunk.toml"
    mode = WriteMode.UPDATE
    driver(mode=mode, toml_file_path=file)
