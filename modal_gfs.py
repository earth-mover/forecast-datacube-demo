#  Everything till the `=======` is required to work, though it can be customized.

from datetime import timedelta

import modal

from src.lib import ReadMode, WriteMode, utcnow
from src.lib_modal import MODAL_FUNCTION_KWARGS
from src.modal_app import applib, driver

app = modal.App("gfs-forecast-ingest")
app.include(applib)  # necessary

# =======


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600, schedule=modal.Cron("30 0,6,12,18 * * *"))
def gfs_update_solar():
    driver(mode=WriteMode.UPDATE, toml_file_path="src/configs/gfs.toml")


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600, schedule=modal.Cron("30 * * * *"))
def gfs_verify_solar():
    driver(mode=ReadMode.VERIFY, toml_file_path="src/configs/gfs.toml")


@app.function(**MODAL_FUNCTION_KWARGS, timeout=3600)
def gfs_backfill():
    file = "src/configs/gfs.toml"
    since = utcnow() - timedelta(days=3)
    till = utcnow() - timedelta(days=1, hours=12)

    driver(mode=WriteMode.BACKFILL, toml_file_path=file, since=since, till=till)


@app.local_entrypoint()
def main():
    gfs_update_solar.remote()
