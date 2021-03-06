from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Callable, Optional, Set

from recording_script_generator.app.exporting import get_tex_path
from recording_script_generator.app.io import *
from recording_script_generator.core import *
from recording_script_generator.core.detection import \
    select_utterances_from_tex_inplace
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection)
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_BATCHES,
                                                DEFAULT_IGNORE,
                                                DEFAULT_MAXTASKSPERCHILD,
                                                DEFAULT_N_JOBS,
                                                DEFAULT_SPLIT_BOUNDARY_MAX_S,
                                                DEFAULT_SPLIT_BOUNDARY_MIN_S)
from text_utils.types import Symbol


def app_select_greedy_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, boundary_min_s: float = DEFAULT_SPLIT_BOUNDARY_MIN_S, boundary_max_s: float = DEFAULT_SPLIT_BOUNDARY_MAX_S, n_jobs: int = DEFAULT_N_JOBS, maxtasksperchild: Optional[int] = DEFAULT_MAXTASKSPERCHILD, chunksize: Optional[int] = None, batches: Optional[int] = DEFAULT_BATCHES, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy duration-based...")
  method = partial(
    select_utterances_through_greedy_duration_inplace,
    n_gram=n_gram,
    minutes=minutes,
    ignore_symbols=ignore_symbols,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    boundary=(boundary_min_s, boundary_max_s),
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  __alter_selection(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_kld_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, boundary_min_s: float = DEFAULT_SPLIT_BOUNDARY_MIN_S, boundary_max_s: float = DEFAULT_SPLIT_BOUNDARY_MAX_S, n_jobs: int = DEFAULT_N_JOBS, maxtasksperchild: Optional[int] = DEFAULT_MAXTASKSPERCHILD, chunksize: Optional[int] = None, batches: Optional[int] = DEFAULT_BATCHES, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD duration-based...")
  method = partial(
    select_utterances_through_kld_duration_inplace,
    n_gram=n_gram,
    minutes=minutes,
    ignore_symbols=ignore_symbols,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    boundary=(boundary_min_s, boundary_max_s),
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  __alter_selection(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


# def app_select_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
#   logger = getLogger(__name__)
#   logger.info("Selecting all...")
#   target = PreparationTarget.REPRESENTATIONS
#   __alter_data(base_dir, corpus_name, in_step_name, target,
#                out_step_name, overwrite, select_all_utterances)


# def app_deselect_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
#   logger = getLogger(__name__)
#   logger.info("Deselecting all...")
#   target = PreparationTarget.REPRESENTATIONS
#   __alter_data(base_dir, corpus_name, in_step_name, target,
#                out_step_name, overwrite, deselect_all_utterances)

def app_select_from_tex(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting from tex...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  step_dir = get_step_dir(corpus_dir, in_step_name)
  tex_path = get_tex_path(step_dir)
  if not tex_path.exists():
    raise Exception("Tex file not found!")

  tex_content = tex_path.read_text()
  method = partial(
    select_utterances_from_tex_inplace,
    tex=tex_content,
  )

  __alter_selection(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def __alter_selection(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str], overwrite: bool, method: Callable[[ReadingPassages, Representations, Selection], None]):
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

  selection, reading_passages, representations, utterance_paths = load_data(in_step_dir)

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

  # utterances are just a copy
  # only on overwrite but the is must be written again: if out_step_dir != in_step_dir:
  save_data(out_step_dir, selection, reading_passages, representations, utterance_paths)
