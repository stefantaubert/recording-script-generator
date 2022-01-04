from argparse import ArgumentParser
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Generator
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from general_utils import get_files_dict
from general_utils.main import get_files_tuples
from tqdm import tqdm


def add_overwrite_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite existing files")


TXT_FILE_TYPE = ".txt"


def get_text_files_tqdm(directory: Path) -> OrderedDictType[str, Path]:
  return OrderedDict(tqdm(get_files_tuples(directory, filetypes={TXT_FILE_TYPE})))


def raise_error_if_directory_not_exists(directory: Path) -> bool:
  if not directory.exists():
    logger = getLogger(__name__)
    logger.error(f"Directory \"{str(directory)}\" was not found!")
    return True

  return False


def raise_error_if_directory_exists_and_not_overwrite(directory: Path, overwrite: bool) -> bool:
  if directory.exists() and not overwrite:
    logger = getLogger(__name__)
    logger.error(f"Directory \"{str(directory)}\" already exists!")
    return True

  return False
