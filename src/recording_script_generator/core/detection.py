from functools import partial
from logging import getLogger
from multiprocessing.pool import RemoteTraceback
from typing import Dict, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.common import log_utterances
from recording_script_generator.core.selection import *
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   Utterance, UtteranceId,
                                                   Utterances)
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


def get_utterance_durations_based_on_symbols(utterances: Utterances, reading_speed_chars_per_s: float) -> Dict[UtteranceId, float]:
  durations = {
    utterance_id: len(symbols) / reading_speed_chars_per_s
    for utterance_id, symbols in utterances.items()
  }
  return durations


def select_utterances_through_kld_duration_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], boundary: DurationBoundary, reading_speed_chars_per_s: float):
  selected = get_selected_utterances(representations, selection)
  deselected = get_deselected_utterances(representations, selection)
  utterance_durations_s = get_utterance_durations_based_on_symbols(
    reading_passages,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  add = get_utterances_through_kld_duration(
    utterances=deselected,
    already_selected_utterances=selected,
    boundary=boundary,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    n_gram=n_gram,
    utterance_durations_s=utterance_durations_s,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_through_kld_iterations_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]):
  deselected = get_deselected_utterances(representations, selection)

  add = get_utterances_through_kld_iterations(
    utterances=deselected,
    ignore_symbols=ignore_symbols,
    iterations=iterations,
    n_gram=n_gram,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_through_greedy_epochs_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]]):
  deselected = get_deselected_utterances(representations, selection)

  add = get_utterances_through_greedy_epochs(
    utterances=deselected,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
    n_gram=n_gram,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_through_greedy_duration_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], mode: SelectionMode, reading_speed_symbols_per_s: float):
  deselected = get_deselected_utterances(representations, selection)
  utterance_durations_s = get_utterance_durations_based_on_symbols(
    reading_passages,
    reading_speed_chars_per_s=reading_speed_symbols_per_s,
  )

  add = get_utterances_through_greedy_duration(
    utterances=deselected,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    n_gram=n_gram,
    utterance_durations_s=utterance_durations_s,
    mode=mode,
  )
  log_and_add_to_selection_inplace(add, selection, reading_passages)


def select_utterances_from_tex_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, tex: str):
  select = get_utterances_from_tex(
    utterances=representations,
    tex=tex,
  )

  log_utterances(reading_passages, select, log_count=None)
  select_selection_inplace(selection, select)
