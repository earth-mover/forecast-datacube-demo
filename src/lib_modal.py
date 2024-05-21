"""Utility functions for modal."""

import modal
from modal import Image

MODAL_IMAGE = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-dev", "curl")
    .pip_install(
        "arraylake",
        "certifi",
        "cfgrib",
        "dask",
        "fsspec",
        "herbie-data",
        "s3fs",
        "xarray",
        "pydantic-core==2.18.2",
        "pydantic==2.7.1",
        "fastapi>=0.108",
    )
    .pip_install("eccodes", "ecmwflibs")
    .env(
        {
            "SSL_CERT_FILE": "/opt/conda/lib/python3.11/site-packages/certifi/cacert.pem",
        }
    )
    .run_commands("python -m eccodes selfcheck")
)


MODAL_FUNCTION_KWARGS = dict(
    image=MODAL_IMAGE,
    secrets=[
        modal.Secret.from_name("earth-mover-aws-secret"),
        modal.Secret.from_name("deepak-arraylake-demos-token"),
    ],
    mounts=[modal.Mount.from_local_python_packages("src")],
)
