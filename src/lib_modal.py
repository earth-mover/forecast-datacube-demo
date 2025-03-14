"""Utility functions for modal."""

from collections.abc import Sequence
from typing import TypedDict

import modal
from modal import Image

MODAL_IMAGE = (
    Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install(
        "arraylake==0.15.1",
        "zarr==2.18.2",
        "certifi",
        "cfgrib",
        "dask==2024.09",
        "fsspec",
        "herbie-data",
        "s3fs",
        "xarray==2024.09",
        "fastapi>=0.108",
        "eccodes==2.37",
        "pyproj",
    )
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
