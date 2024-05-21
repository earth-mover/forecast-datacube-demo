#  Everything till the `=======` is required to work, though it can be customized.

from datetime import datetime, timedelta

import modal

from modal_app import applib, driver
from src.lib import WriteMode
from src.lib_modal import MODAL_FUNCTION_KWARGS

app = modal.App("hrrr-forecast-ingest")
app.include(applib)  # necessary


@app.local_entrypoint()
def main(mode: str, toml_file: str, since: str, till: str | None = None):
    if mode != "backfill":
        raise ValueError("Only mode='backfill' is allowed.")

    driver(mode=WriteMode.BACKFILL, toml_file_path=toml_file, since=since, till=till)


# =======


@app.function(**MODAL_FUNCTION_KWARGS)
def hrrr_backfill():
    """Run this "backfill" function wtih `modal run modal_hrrr.py::hrrr_backfill`."""
    file = "src/configs/hrrr.toml"
    mode = WriteMode.BACKFILL
    since = datetime.utcnow() - timedelta(days=3)
    till = datetime.utcnow() - timedelta(days=1, hours=12)

    driver(mode=mode, toml_file_path=file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, schedule=modal.Cron("57 * * * *"))
def hrrr_update_solar():
    """
    *Deploy* this :update" function wtih `modal deploy modal_hrrr.py --name hrrr_update_solar`.
    """
    driver(mode=WriteMode.UPDATE, toml_file_path="src/configs/hrrr.toml")


@app.function(**MODAL_FUNCTION_KWARGS)
def hrrr_backfill_rechunk():
    file = "src/configs/hrrr-demo.toml"
    mode = WriteMode.BACKFILL
    since = datetime.utcnow() - timedelta(days=3)
    till = datetime.utcnow() - timedelta(days=1, hours=12)

    driver(mode=mode, toml_file_path=file, since=since, till=till)


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600, schedule=modal.Cron("57 * * * *"))
def hrrr_update_rechunk():
    driver(mode=WriteMode.UPDATE, toml_file_path="src/configs/hrrr-demo.toml")
