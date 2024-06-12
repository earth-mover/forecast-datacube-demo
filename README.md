# Analysis-Ready Weather Forecast Data Cubes in Zarr

This is a model for how to build a data pipeline that ingests and transforms weather forecast data distributed as GRIB files to an analysis-ready cloud-optimized Zarr data cube using a serverless pipeline.

Currently two models are supported : GFS and HRRR, extensions to other models should be easy™.

> [!WARNING]
> This repo is for demonstration purposes only. It does not aspire to be a maintained package.
> If you want to build on top of it, fork this repo and modify it to your needs.

The code runs in one of two modes:
1. `backfill` : Initialize a new Zarr store or Arraylake repo. Ingest data for the time period between `since` and (optionally) `till`.
1. `update` : Designed to run as a scheduled Cron job, this mode brings a Zarr store up-to-date with the latest available data.

Modal Cron Jobs cannot take arguments, so any configuration must be shipped up when running `modal deploy`. To configure the data pipeline
set up a TOML file in `src/configs/`. The location is arbitrary, but it's nice to keep it all in one place.

## Execution
``` sh
# Drive from the command line.
modal run modal_hrrr.py --mode "backfill" --since "2024-05-15"  --toml-file src/configs/hrrr-demo.toml

# Directly specify parameters `since`, `till`, `toml_file_path` in hrrr_backfill.
modal run modal_hrrr.py::hrrr_backfill

# Set up a repeating cron job to update the store.
modal deploy modal_hrrr.py::hrrr_update_solar
```

## Configuration

Details of the pipeline are configured using a TOML file.

```
# Format
# ------
[arbitrary_job_name]
## Three Herbie Parameters
model = string
product = string
search = string
## Two Zarr store Parameters
store = string  e.g. s3://my-bucket/store.zarr
zarr_group = string e.g. "sfc/fcst"
## Schema
chunks = {string: int}, integer chunk size for dimension
renames = {str: str}, mapping from variable name in inventory to actual name when read with cfgrib
```

Here's a concrete example:
```toml
[job1]
model = "hrrr"
product = "sfc"
searches = [
  "(?:TMP|RH):2 m above ground|(?:GUST|DSWRF|PRATE):surface|TCDC:entire atmosphere",
]
store = "arraylake://earthmover-demos/hrrr"
zarr_group =  "solar/"
chunks = {x = 360, y = 120, time = 1, step = 19}
renames = {TMP="t2m", RH="r2", TCDC="tcc"}
```

- Set `model` and `product` as necessary to have `herbie` find the right variables.
- Set the output location using `store` and `zarr_group` within the store. For example, `store` can be `s3://my-bucket/forecast-datacube.zarr`.
- `chunks` specify chunking for the Zarr arrays.
- To set up the `searches` we recommend iterating interactively with `FastHerbie.inventory(search=searches)` and making sure you see all data that's needed.
- The `renames` field is harder. It is necessary because `searches` (and `herbie`) will only know variable names as written in the `.idx` sidecar files.
  The variable names are *not* necessarily preserved when reading those GRIB files with `cfgrib`.
  Annoyingly, we do not know *a priori* what names `cfgrib` will choose to assign.
  Again the best approach is to iterate in a notebook.
  Alternatively, simply run the pipeline with `renames={}`, and an error will raised suggesting what to set.

For more examples see `src/configs/`.

## Organization

- `hrrr-cube.ipynb` : Notebook demonstrating analysis with the HRRR data cube.
- `modal_app.py` : Core functions annotated to run with Modal.
- Model-specific functions.
  - `modal_hrrr.py`, `modal_gfs.py`
  - These are simple specializations for a couple of models: HRRR and GFS. They have been separated out for convenience.
- `src/`:
  1. `lib.py`: Core data structures and utilities. The most important data structure is `ForecastModel`. This is the base class that allows specialization to a specific model.
  1. `gfs.py` : Contains `GFS`, a subclass of `ForecastModel`, specialized for GFS output.
  1. `hrrr.py` : Contains `HRRR`, a subclass of `ForecastModel`, specialized for HRRR output.


## Sharp edges

1. Make sure that the `search` string returns what you want. It is a good idea to use ``FastHerbie.inventory(search_string)`` to double check.
1. `cfgrib` likes to rename variables so what's in the dataset doesn't match what's in the `search` string. Please specify `renames` as a dictionary that maps variable name in the GRIB inventory file to variable name set by `cfgrib`.
1. Multiple `searches` are not supported yet.
1. The combination of `zarr_store` & `group` cannot be repeated in the TOML file.
1. Modal functions with schedules cannot take arguments. So any configuration `toml` files must be uploaded during `modal deploy` by bundling them in `src/configs/`.
   - See https://herbie.readthedocs.io/en/stable/user_guide/tutorial/search.html for more

## Acknowledgments

- [Herbie](https://herbie.readthedocs.io/en/stable/)
- [cfgrib](https://github.com/ecmwf/cfgrib)
