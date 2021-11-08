from enum import IntEnum
from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Callable, Optional

from recording_script_generator.app.io import *
from recording_script_generator.core import *
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Utterances)
from recording_script_generator.core.utterances.transformation.arpa_to_ipa import \
    map_utterances_from_arpa_to_ipa_inplace
from recording_script_generator.core.utterances.transformation.change_ipa import \
    change_utterances_ipa_inplace
from recording_script_generator.core.utterances.transformation.change_text import \
    change_utterances_text_inplace
from recording_script_generator.core.utterances.transformation.eng_to_arpa import \
    convert_utterances_from_eng_to_arpa_inplace
from recording_script_generator.core.utterances.transformation.normalize import \
    normalize_utterances_inplace


class Target(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1


def get_target_utterances(reading_passages: ReadingPassages, representations: Representations, target: Target) -> Utterances:
  if target == Target.READING_PASSAGES:
    return reading_passages
  elif target == Target.REPRESENTATIONS:
    return representations
  assert False


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize_utterances_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_convert_to_arpa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to ARPA...")
  method = partial(
    convert_utterances_from_eng_to_arpa_inplace,
    n_jobs=n_jobs, chunksize=chunksize, maxtasksperchild=maxtasksperchild,

  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_map_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Mapping to IPA...")
  method = partial(
    map_utterances_from_arpa_to_ipa_inplace,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_change_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_utterances_ipa_inplace,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )

  __alter_utterances(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_change_text(base_dir: Path, corpus_name: str, in_step_name: str, target: Target, remove_space_around_punctuation: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing text...")
  method = partial(
    change_utterances_text_inplace,
    remove_space_around_punctuation=remove_space_around_punctuation,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
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

  selection, reading_passages, representations = load_data(in_step_dir)
  utterances = get_target_utterances(reading_passages, representations, target)

  start = perf_counter()
  method(utterances)
  end = perf_counter()
  duration_min = (end - start) / 60
  logger.info(f"Total operation duration: {duration_min:.2f}min.")

  if out_step_dir.exists():
    assert overwrite
    logger.info("Removing existing out dir...")
    rmtree(out_step_dir)
    logger.info("Done.")
  out_step_dir.mkdir(parents=True, exist_ok=True)

  # selection is just a copy
  save_data(out_step_dir, selection, reading_passages, representations)
