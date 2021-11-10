from enum import IntEnum
from functools import partial
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from recording_script_generator.app.io import *
from recording_script_generator.core import *
from recording_script_generator.core.types import Utterances
from recording_script_generator.core.utterances.transformation import (
    change_utterances_ipa_inplace, change_utterances_text_inplace,
    convert_to_symbols_inplace, convert_utterances_from_eng_to_arpa_inplace,
    map_utterances_from_arpa_to_ipa_inplace, normalize_utterances_inplace)


class Target(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1


def load_target_utterances(target: Target, step_dir: Path) -> Utterances:
  if target == Target.READING_PASSAGES:
    return load_reading_passages(step_dir)
  elif target == Target.REPRESENTATIONS:
    return load_representations(step_dir)
  assert False


def save_target_utterances(target: Target, step_dir: Path, utterances: Utterances) -> None:
  if target == Target.READING_PASSAGES:
    save_reading_passages(step_dir, utterances)
  elif target == Target.REPRESENTATIONS:
    save_representations(step_dir, utterances)
  else:
    assert False


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize_utterances_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_convert_to_arpa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to ARPA...")
  method = partial(
    convert_utterances_from_eng_to_arpa_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_map_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Mapping to IPA...")
  method = partial(
    map_utterances_from_arpa_to_ipa_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_convert_to_symbols(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to symbols...")
  method = partial(
    convert_to_symbols_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_change_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_utterances_ipa_inplace,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_change_text(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, remove_space_around_punctuation: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing text...")
  method = partial(
    change_utterances_text_inplace,
    remove_space_around_punctuation=remove_space_around_punctuation,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def __alter_utterances(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, out_step_name: Optional[str], overwrite: bool, method: Callable[[Utterances], None]):
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

  utterances = load_target_utterances(target, in_step_dir)

  start = perf_counter()
  method(utterances)
  end = perf_counter()
  duration_min = (end - start) / 60
  logger.info(f"Total operation duration: {duration_min:.2f}min.")

  # if out_step_dir.exists():
  #   assert overwrite
  #   logger.info("Removing existing out dir...")
  #   rmtree(out_step_dir)
  #   logger.info("Done.")
  out_step_dir.mkdir(parents=True, exist_ok=True)

  # selection is just a copy
  save_target_utterances(target, out_step_dir, utterances)
  del utterances

  if in_step_dir != out_step_dir:
    logger.info("Copying other data from input folder.")
    copy_selection(in_step_dir, out_step_dir)
    if target == Target.READING_PASSAGES:
      copy_representations(in_step_dir, out_step_dir)
    elif target == Target.REPRESENTATIONS:
      copy_reading_passages(in_step_dir, out_step_dir)
  logger.info("Done.")
