from functools import partial
from logging import getLogger
from math import inf
from multiprocessing.pool import RemoteTraceback
from sys import getsizeof
from time import perf_counter
from typing import Dict, Optional, Set, cast

from ordered_set import OrderedSet
from recording_script_generator.core.detection import (
    get_utterance_durations_based_on_symbols,
    select_utterances_through_kld_duration_inplace)
from recording_script_generator.core.inspection import (
    remove_duplicate_utterances_inplace,
    remove_utterances_with_acronyms_inplace,
    remove_utterances_with_custom_word_counts,
    remove_utterances_with_non_dictionary_words,
    remove_utterances_with_proper_names, remove_utterances_with_undesired_text,
    remove_utterances_with_unfrequent_words)
from recording_script_generator.core.selection import *
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   Utterance, UtteranceId,
                                                   Utterances)
from recording_script_generator.core.utterances import *
from tqdm import tqdm


def handle_error(exception: Exception) -> None:
  logger = getLogger(__name__)
  #tb = sys.exc_info()
  # traceback.print_stack()
  # print(traceback.format_exc())
  logger.exception(exception)
  remote_traceback = cast(RemoteTraceback, exception.__cause__)
  logger.info(remote_traceback.tb)
  pass


def remove_inplace(remove: Set[UtteranceId], reading_passages: ReadingPassages, representations: Representations, selection: Selection) -> None:
  log_utterances(representations, remove, log_count=10)
  remove_from_selection_inplace(selection, remove)
  remove_from_utterances_inplace(reading_passages, remove)
  remove_from_utterances_inplace(representations, remove)


def log_utterances(utterances: Utterances, selection: Set[UtteranceId], log_count: Optional[int] = 10) -> None:
  logger = getLogger(__name__)
  if log_count is None:
    log_count = len(selection)
  logger.info("Utterances:")
  for utterance_id in list(sorted(selection))[:log_count]:
    utterance_str = ''.join(utterances[utterance_id])
    logger.info(f"- \"{utterance_str}\" ({utterance_id}),")
  if log_count < len(selection):
    logger.info(f"- and {len(selection) - log_count} further utterance(s)...")


def do_pipeline_prepare_inplace(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: Optional[int]) -> None:
  kwargs = {
    "reading_passages": reading_passages,
    "representations": representations,
    "selection": selection,
  }

  mp_kwargs = {
    "n_jobs": n_jobs,
    "chunksize": chunksize,
    "maxtasksperchild": maxtasksperchild,
  }

  # Remove acronyms
  remove_utterances_with_acronyms_inplace(
    min_acronym_len=3,
    **(kwargs | mp_kwargs),
  )

  # Normalize
  normalize_utterances_inplace(
    reading_passages,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )
  normalize_utterances_inplace(
    representations,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )

  # Remove duplicates
  remove_duplicate_utterances_inplace(
    **kwargs,
  )

  # Remove infrequent words
  remove_utterances_with_unfrequent_words(
    min_occurrence_count=2,
    **(kwargs | mp_kwargs),
  )

  # Remove proper names
  remove_utterances_with_proper_names(
    min_occurrence_count=2,
    **(kwargs | mp_kwargs),
  )

  # Remove undesired text
  undesired = set("/ \\ - : @ ; * % \" ( ) [ ] { } quote oswald ye hath pp.".split(" "))
  remove_utterances_with_undesired_text(
    undesired=undesired,
    **(kwargs | mp_kwargs),
  )

  # Remove utterances with < 3 words
  remove_utterances_with_custom_word_counts(
    min_count=3,
    max_count=None,
    **(kwargs | mp_kwargs),
  )

  # Remove utterances that contain non-dictionary words
  remove_utterances_with_non_dictionary_words(
    max_unknown_word_count=0,
    **(kwargs | mp_kwargs),
  )

  # Convert to ARPA
  convert_utterances_from_eng_to_arpa_inplace(
    representations,
    n_jobs=n_jobs, chunksize=chunksize,
  )

  # Remove undesired text
  undesired = {"'"}
  remove_utterances_with_undesired_text(
    undesired=undesired,
    **(kwargs | mp_kwargs),
  )

  # Map ARPA to IPA
  map_utterances_from_arpa_to_ipa_inplace(
    representations,
    n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=chunksize,
  )


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


def update_utterances(add: OrderedSet[UtteranceId], selected: Representations, deselected: Representations, selection: Selection, reading_passages: ReadingPassages):
  log_utterances(reading_passages, add, log_count=None)
  add_to_selection_inplace(selection, add)
  add_to_utterances_inplace(selected, add, deselected)
  remove_from_utterances_inplace(deselected, add)


def do_pipeline_select(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: int):
  utterance_durations_s = get_utterance_durations_based_on_symbols(
    reading_passages,
    reading_speed_symbols_per_s=14
  )

  selected = get_selected_utterances(representations, selection)
  deselected = get_deselected_utterances(representations, selection)

  update_utterances_func = partial(
    update_utterances,
  )

  add = get_utterances_through_kld_duration(
    utterances=deselected,
    already_selected_utterances=selected,
    boundary=(8, inf),
    ignore_symbols=" ",
    minutes=1,  # 10
    n_gram=1,
    utterance_durations_s=utterance_durations_s,
  )
  update_utterances_func(add)

  add = get_utterances_through_kld_duration(
    utterances=deselected,
    already_selected_utterances=selected,
    boundary=(4, 8),
    ignore_symbols=" ",
    minutes=1,  # 13
    n_gram=1,
    utterance_durations_s=utterance_durations_s,
  )
  update_utterances_func(add)

  add = get_utterances_through_kld_duration(
    utterances=deselected,
    already_selected_utterances=selected,
    boundary=(0, 4),
    ignore_symbols=" ",
    minutes=1,  # 10
    n_gram=1,
    utterance_durations_s=utterance_durations_s,
  )
  update_utterances_func(add)

  remove_from_utterances_inplace(reading_passages, remove=deselected.keys())
  remove_from_utterances_inplace(representations, remove=deselected.keys())
  selection_contains_all_keys = len(selection - representations.keys()) == 0
  assert selection_contains_all_keys
