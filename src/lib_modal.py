"""Utility functions for modal."""

from collections.abc import Sequence
from typing import TypedDict

import modal
from modal import Image

MODAL_IMAGE = (
    # https://linear.app/earthmover/issue/EAR-1067/python-312-threading-issue-at-shutdown-time
    Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        ". ~/.cargo/env",
        "echo $PATH && cargo -V",
    )
    .pip_install(
        "arraylake==0.14.0",
        # "icechunk==0.1.1",
        "git+https://github.com/earth-mover/icechunk.git@push-nltsokxuurpz#subdirectory=icechunk-python/",
        "certifi",
        "cfgrib",
        "dask",
        "fsspec",
        "herbie-data",
        "s3fs",
        "xarray",
        "fastapi>=0.108",
        "eccodes==2.37",
        "pyproj",
        "zarr==3.0.2",
        "ipdb",
    )
    # .env({"PYTHONASYNCIODEBUG": "1"})
    .add_local_python_source("src")
)


class ModalKwargs(TypedDict):
    image: modal.Image
    secrets: Sequence[modal.Secret]


MODAL_FUNCTION_KWARGS: ModalKwargs = dict(
    image=MODAL_IMAGE,
    secrets=[
        modal.Secret.from_name("ryan-aws-secret"),
        modal.Secret.from_name("deepak-arraylake-demos-token"),
    ],
)
