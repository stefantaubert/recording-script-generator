from logging import getLogger
from multiprocessing.pool import RemoteTraceback
from typing import Optional, Set

from recording_script_generator.core.common import log_utterances
from recording_script_generator.core.selection import *
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   UtteranceId)
from recording_script_generator.core.utterances import *


def __remove_inplace(remove: Set[UtteranceId], reading_passages: ReadingPassages, representations: Representations, selection: Selection) -> None:
  log_utterances(representations, remove, log_count=10)
  remove_from_selection_inplace(selection, remove)
  remove_from_utterances_inplace(reading_passages, remove)
  remove_from_utterances_inplace(representations, remove)


def remove_utterances_with_acronyms_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, min_acronym_len: int, n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_acronyms(
    utterances=representations,
    min_acronym_len=min_acronym_len,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_duplicate_utterances_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection):
  remove = get_duplicate_utterances(
    utterances=representations,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_utterances_with_unfrequent_words(reading_passages: ReadingPassages, representations: Representations, selection: Selection, min_occurrence_count: int, n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_unfrequent_words(
    utterances=representations,
    min_occurrence_count=min_occurrence_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_utterances_with_proper_names(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_proper_names(
    utterances=representations,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_utterances_with_undesired_text(reading_passages: ReadingPassages, representations: Representations, selection: Selection, undesired: Set[str], n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_undesired_text(
    utterances=representations,
    undesired=undesired,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_utterances_with_custom_word_counts(reading_passages: ReadingPassages, representations: Representations, selection: Selection, min_count: Optional[int], max_count: Optional[int], n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_custom_word_counts(
    utterances=representations,
    min_count=min_count,
    max_count=max_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_utterances_with_non_dictionary_words(reading_passages: ReadingPassages, representations: Representations, selection: Selection, max_unknown_word_count: int, n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]):
  remove = get_utterances_with_non_dictionary_words(
    utterances=representations,
    max_unknown_word_count=max_unknown_word_count,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  __remove_inplace(remove, reading_passages, representations, selection)


def remove_deselected_utterances(reading_passages: ReadingPassages, representations: Representations, selection: Selection) -> None:
  remove = representations.keys() - selection
  __remove_inplace(remove, reading_passages, representations, selection)
