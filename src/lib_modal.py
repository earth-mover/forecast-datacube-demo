"""Utility functions for modal."""

from collections.abc import Sequence
from typing import TypedDict

import modal
from modal import Image

MODAL_IMAGE = (
    Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install(
        "arraylake>=0.13.3",
        "icechunk >= 0.1.0a12",
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
