"""Utility functions for modal."""

from collections.abc import Sequence
from typing import TypedDict

import modal
from modal import Image

MODAL_IMAGE = (
    # https://linear.app/earthmover/issue/EAR-1067/python-312-threading-issue-at-shutdown-time
    Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    # .apt_install("curl", "git")
    # .env({"PATH": "/root/.cargo/bin:$PATH"})
    # .run_commands(
    #     "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    #     ". ~/.cargo/env",
    #     "echo $PATH && cargo -V",
    # )
    .uv_sync()
    # .env({"PYTHONASYNCIODEBUG": "1"})
    .add_local_python_source("src")
    .add_local_dir("src/configs", remote_path="/root/src/configs")
)


class ModalKwargs(TypedDict):
    image: modal.Image
    secrets: Sequence[modal.Secret]


MODAL_FUNCTION_KWARGS: ModalKwargs = dict(
    image=MODAL_IMAGE,
    secrets=[
        modal.Secret.from_name("deepak-earthmover-public-token"),
        modal.Secret.from_name("deepak-earthmover-integration-token"),
    ],
)
