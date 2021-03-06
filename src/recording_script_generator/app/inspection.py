from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Callable, Optional, Set

from recording_script_generator.app.io import *
from recording_script_generator.core import *
from recording_script_generator.core.types import Representations, UtteranceId
from text_utils.types import Symbol


def app_remove_utterances_with_acronyms(base_dir: Path, corpus_name: str, in_step_name: str, min_acronym_len: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with acronyms...")
  method = partial(
    remove_utterances_with_acronyms_inplace,
    min_acronym_len=min_acronym_len,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name,
               overwrite, method)


def app_remove_duplicate_utterances(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing duplicate utterances...")
  __alter_data(base_dir, corpus_name, in_step_name,
               out_step_name, overwrite, remove_duplicate_utterances_inplace)


def app_remove_utterances_with_proper_names(base_dir: Path, corpus_name: str, in_step_name: str, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with proper names...")
  method = partial(
    remove_utterances_with_proper_names,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name,
               overwrite, method)


def app_remove_utterances_with_undesired_sentence_lengths(base_dir: Path, corpus_name: str, in_step_name: str, min_count: Optional[int], max_count: Optional[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with undesired word counts...")
  method = partial(
    remove_utterances_with_custom_word_counts,
    min_count=min_count,
    max_count=max_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_unknown_words(base_dir: Path, corpus_name: str, in_step_name: str, max_unknown_word_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with to many unknown words...")
  method = partial(
    remove_utterances_with_non_dictionary_words,
    max_unknown_word_count=max_unknown_word_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_too_seldom_words(base_dir: Path, corpus_name: str, in_step_name: str, min_occurrence_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with too seldom words...")
  method = partial(
    remove_utterances_with_unfrequent_words,
    min_occurrence_count=min_occurrence_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_undesired_text(base_dir: Path, corpus_name: str, in_step_name: str, undesired: Set[Symbol], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing undesired text...")
  method = partial(
    remove_utterances_with_undesired_text,
    undesired=undesired,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize, batches=batches,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_deselected(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing deselected...")
  method = partial(
    remove_deselected_utterances,
  )

  __alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def __alter_data(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str], overwrite: bool, method: Callable[[Representations, Selection], Set[UtteranceId]]):
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

  selection = load_selection(in_step_dir)
  representations = load_representations(in_step_dir)
  old_count = len(representations)

  start = perf_counter()
  # reading passages as target is not planned
  remove = method(representations, selection)
  end = perf_counter()
  duration_min = (end - start) / 60
  logger.info(f"Total operation duration: {duration_min:.2f}min.")

  final_count = len(representations)
  removed_count = old_count - final_count
  if old_count > 0:
    logger.info(
      f"Removed {removed_count} of {old_count} utterances ({removed_count/old_count*100:.2f}%) and obtained {final_count} utterances.")

  # if out_step_dir.exists():
  #   assert overwrite
  #   logger.info("Removing existing out dir...")
  #   rmtree(out_step_dir)
  #   logger.info("Done.")
  out_step_dir.mkdir(parents=True, exist_ok=True)

  save_selection(out_step_dir, selection)
  del selection
  save_representations(out_step_dir, representations)
  del representations

  changed_something = len(remove) > 0
  if changed_something:
    # for memory reasons, remove the items at the end
    logger.info("Applying changes also to reading passages...")
    reading_passages = load_reading_passages(in_step_dir)
    remove_from_utterances_inplace(reading_passages, remove)
    save_reading_passages(out_step_dir, reading_passages)
  else:
    if in_step_dir != out_step_dir:
      logger.info("Copying reading passages...")
      copy_reading_passages(in_step_dir, out_step_dir)
  logger.info("Done.")
