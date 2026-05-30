# coding=utf-8

from pathlib import Path, PosixPath
from typing import Union, List

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def str2path(*args: Union[Path, str]) -> List[Path]:
    arg_paths: List[Path] = []
    for arg in args:
        if isinstance(arg, str):
            arg_paths.append(Path(arg).resolve())
        elif isinstance(arg, Path):
            arg_paths.append(arg.resolve())
        else:
            logger.error(f"{arg} not a valid path")
    return arg_paths


def assert_pcap(file: PosixPath) -> bool:
    assert file.is_file() and file.suffix == ".pcap", logger.exception(f"{file} is not a pcap file")
    return True


def is_pcap(file: PosixPath) -> bool:
    if file.is_file() and file.suffix == ".pcap":
        return True


def base_name(path: PosixPath) -> PosixPath:
    if path.is_file():
        full_path = path.resolve()
        return full_path.parent
    elif path.is_dir():
        return path.resolve()
    else:
        raise RuntimeError(f"not a valid file or path {path}")


