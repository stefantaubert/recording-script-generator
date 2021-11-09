from logging import getLogger
from pathlib import Path
from shutil import copyfile
from sys import getsizeof
from time import perf_counter
from typing import Tuple

from general_utils import load_obj, save_obj
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection)

READING_PASSAGES_DATA_FILE = "reading_passages.pkl"
REPRESENTATIONS_DATA_FILE = "representations.pkl"
SELECTION_DATA_FILE = "selection.pkl"


def get_corpus_dir(base_dir: Path, corpus_name: str) -> Path:
  return base_dir / corpus_name


def get_step_dir(corpus_dir: Path, step_name: str) -> Path:
  return corpus_dir / step_name


def load_reading_passages(step_dir: Path) -> ReadingPassages:
  logger = getLogger(__name__)
  logger.info("Loading reading passages...")
  start = perf_counter()
  res = load_obj(step_dir / READING_PASSAGES_DATA_FILE)
  logger.info(f"Done. Loaded {len(res)} reading passages in {perf_counter() - start:.2f}s.")
  if __debug__:
    logger.info("Calculating size in memory...")
    size = getsizeof(res)
    #from pympler import asizeof
    #size = asizeof.asizeof(res)
    logger.info(f"Size of reading passages in memory: {size/1024**3:.2f} Gb.")
  return res


def load_representations(step_dir: Path) -> Representations:
  logger = getLogger(__name__)
  logger.info("Loading representations...")
  start = perf_counter()
  res = load_obj(step_dir / REPRESENTATIONS_DATA_FILE)
  logger.info(f"Done. Loaded {len(res)} representations in {perf_counter() - start:.2f}s.")

  if __debug__:
    logger.info("Calculating size in memory...")
    size = getsizeof(res)
    #from pympler import asizeof
    #size = asizeof.asizeof(res)
    logger.info(f"Size of representations in memory: {size/1024**3:.2f} Gb.")
  return res


def load_selection(step_dir: Path) -> Selection:
  res = load_obj(step_dir / SELECTION_DATA_FILE)
  return res


def save_reading_passages(step_dir: Path, reading_passages: ReadingPassages) -> None:
  logger = getLogger(__name__)
  logger.info("Saving reading passages...")
  start = perf_counter()
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / READING_PASSAGES_DATA_FILE,
    obj=reading_passages,
  )
  logger.info(
    f"Done. Saved {len(reading_passages)} reading passages in {perf_counter() - start:.2f}s.")

  return None


def save_representations(step_dir: Path, representations: Representations) -> None:
  logger = getLogger(__name__)
  logger.info("Saving representations...")
  start = perf_counter()
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / REPRESENTATIONS_DATA_FILE,
    obj=representations,
  )
  logger.info(
    f"Done. Saved {len(representations)} representations in {perf_counter() - start:.2f}s.")

  return None


def copy_selection(in_step_dir: Path, out_step_dir: Path) -> None:
  copyfile(in_step_dir / SELECTION_DATA_FILE, out_step_dir / SELECTION_DATA_FILE)

def copy_representations(in_step_dir: Path, out_step_dir: Path) -> None:
  copyfile(in_step_dir / REPRESENTATIONS_DATA_FILE, out_step_dir / REPRESENTATIONS_DATA_FILE)


def copy_reading_passages(in_step_dir: Path, out_step_dir: Path) -> None:
  copyfile(in_step_dir / READING_PASSAGES_DATA_FILE, out_step_dir / READING_PASSAGES_DATA_FILE)


def save_selection(step_dir: Path, selection: Selection) -> None:
  logger = getLogger(__name__)
  logger.info("Saving selection...")
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / SELECTION_DATA_FILE,
    obj=selection,
  )
  logger.info("Done.")

  return None


def save_data(step_dir: Path, selection: Selection, reading_passages: ReadingPassages, representations: Representations) -> None:
  save_selection(step_dir, selection)
  save_reading_passages(step_dir, reading_passages)
  save_representations(step_dir, representations)


def load_data(step_dir: Path) -> Tuple[Selection, ReadingPassages, Representations]:
  selection = load_selection(step_dir)
  reading_passages = load_reading_passages(step_dir)
  representations = load_representations(step_dir)
  return selection, reading_passages, representations
