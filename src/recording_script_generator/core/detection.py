from functools import partial
from logging import getLogger
from multiprocessing.pool import RemoteTraceback
from typing import Dict, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.common import log_utterances
from recording_script_generator.core.selection import *
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   UtteranceId,Utterances,
                                                   get_utterance_duration_s)
from recording_script_generator.core.utterances import *
from text_selection.selection import SelectionMode
from text_selection.utils import DurationBoundary
from text_utils.types import Symbol
from tqdm import tqdm


def log_and_add_to_selection_inplace(add: OrderedSet[UtteranceId], selection: Selection, reading_passages: ReadingPassages):
  log_utterances(reading_passages, add, log_count=None)
  add_to_selection_inplace(selection, add)


def get_utterances_from_selection(utterances: Utterances, selection: OrderedSet[UtteranceId]) -> Utterances:
  result = Utterances({k: utterances[k] for k in tqdm(selection)})
  result.language = utterances.language
  result.symbol_format = utterances.symbol_format
  return result


def get_selected_utterances(utterances: Utterances, selection: Selection) -> Utterances:
  return get_utterances_from_selection(utterances, selection)


def get_deselected_utterances(utterances: Utterances, selection: Selection) -> Utterances:
  deselected = OrderedSet(utterances.keys() - selection)
  return get_utterances_from_selection(utterances, deselected)


def get_utterance_durations_based_on_utterances(utterances: Utterances, keys: OrderedSet[int], reading_speed_chars_per_s: float) -> Dict[UtteranceId, float]:
  durations = {
    utterance_id: get_utterance_duration_s(utterances[utterance_id], reading_speed_chars_per_s)
    for utterance_id in tqdm(keys)
  }
  return durations


def select_utterances_through_kld_duration_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], boundary: DurationBoundary, reading_speed_chars_per_s: float, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]):
  deselected = OrderedSet(representations.keys() - selection)
  logger = getLogger(__name__)
  logger.info("Getting durations...")
  deselected_durations_s = get_utterance_durations_based_on_utterances(
    reading_passages,
    keys=deselected,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  add = get_utterances_through_kld_duration(
    utterances=representations,
    deselected=deselected,
    selected=selection,
    boundary=boundary,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    n_gram=n_gram,
    deselected_durations_s=deselected_durations_s,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_through_greedy_duration_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], boundary: DurationBoundary, reading_speed_chars_per_s: float, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]):
  deselected = OrderedSet(representations.keys() - selection)
  logger = getLogger(__name__)
  logger.info("Getting durations...")
  deselected_durations_s = get_utterance_durations_based_on_utterances(
    reading_passages,
    keys=deselected,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  add = get_utterances_through_greedy_duration(
    utterances=representations,
    deselected=deselected,
    selected=selection,
    boundary=boundary,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    n_gram=n_gram,
    deselected_durations_s=deselected_durations_s,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_from_tex_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, tex: str):
  select = get_utterances_from_tex(
    utterances=representations,
    tex=tex,
  )

  log_utterances(reading_passages, select, log_count=None)
  select_selection_inplace(selection, select)
