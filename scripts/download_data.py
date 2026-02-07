#!/usr/bin/env python3
"""Download all datasets for TermoGrid AI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download_datasets import download_all
from loguru import logger

if __name__ == "__main__":
    logger.info("Downloading TermoGrid AI datasets...")
    paths = download_all()
    for name, p in paths.items():
        logger.info(f"  {name}: {p}")
