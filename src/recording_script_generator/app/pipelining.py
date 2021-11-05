from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Callable, Optional, Set

from recording_script_generator.app.io import *
from recording_script_generator.core import *
from recording_script_generator.core.preprocessing_pipeline import (
    do_pipeline_prepare_inplace, do_pipeline_select)
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection)


def app_do_pipeline_prepare(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: str, chunksize: int, maxtasksperchild: Optional[int], n_jobs: int, overwrite: bool) -> None:
  method = partial(
    do_pipeline_prepare_inplace,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_do_pipeline_select(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: str, chunksize: int, maxtasksperchild: Optional[int], n_jobs: int, overwrite: bool) -> None:
  method = partial(
    do_pipeline_select,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def __alter_data(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str], overwrite: bool, method: Callable[[ReadingPassages, Representations, Selection], None]):
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
  if out_step_name is None:
    out_step_dir = in_step_dir
  else:
    out_step_dir = get_step_dir(corpus_dir, out_step_name)

  if not corpus_dir.exists():
    logger.info("Corpus dir does not exists.")
    return
  if not in_step_dir.exists():
    logger.info("In dir not exists.")
    return
  if out_step_dir.exists() and not overwrite:
    logger.info("Already exists.")
    return

  selection, reading_passages, representations = load_data(in_step_dir)

  start = perf_counter()
  method(reading_passages, representations, selection)
  end = perf_counter()
  duration_min = (end - start) / 60
  logger.info(f"Total operation duration: {duration_min:.2f}min.")

  if out_step_dir.exists():
    assert overwrite
    logger.info("Removing existing out dir...")
    rmtree(out_step_dir)
    logger.info("Done.")
  out_step_dir.mkdir(parents=True, exist_ok=True)

  save_data(out_step_dir, selection, reading_passages, representations)
