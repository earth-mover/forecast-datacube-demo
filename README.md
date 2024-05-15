This is not a production tool. Please adapt and fix as needed.

## Sharp edges

1. Make sure that the `search` string returns what you want. It is a good idea to use ``FastHerbie.inventory(search_string)`` to double check.
1. `cfgrib` likes to rename variables so what's in the dataset doesn't match what's in the `search` string. Please specify `renames` as a dictionary that maps variable name in the GRIB inventory file to variable name set by `cfgrib`.
1. Multiple `searches` are not supported yet.
1. The combination of `zarr_store` & `group` cannot be repeated in the TOML file.
1. Modal functions with schedules cannot take arguments. So any configuration `toml` files must be uploaded during `modal deploy` by bundling them in `src/configs/`.
   - See https://herbie.readthedocs.io/en/stable/user_guide/tutorial/search.html for more

