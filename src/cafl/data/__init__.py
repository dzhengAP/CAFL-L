"""Data module."""

from .dataset import (
    download_text,
    normalize_text,
    build_charset,
    encode_text,
    split_clients,
    prepare_shakespeare_data,
    batchify
)

__all__ = [
    "download_text",
    "normalize_text",
    "build_charset",
    "encode_text",
    "split_clients",
    "prepare_shakespeare_data",
    "batchify"
]
