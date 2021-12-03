from logging import getLogger
from typing import Dict, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from text_selection import greedy_uniform_ngrams_seconds_with_preselection_perf
from text_selection.utils import DurationBoundary
from text_utils.types import Symbol


def get_utterances_through_greedy_duration(utterances: Utterances, selected: OrderedSet[int], deselected: OrderedSet[int], deselected_durations_s: Dict[UtteranceId, float], n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[UtteranceId]:
  seconds = minutes * 60
  newly_selected = greedy_uniform_ngrams_seconds_with_preselection_perf(
    data=utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=seconds,
    preselection_keys=selected,
    select_from_durations_s=deselected_durations_s,
    select_from_keys=deselected,
    duration_boundary=boundary,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  newly_selected_duration_s = sum(
    duration_s
    for utterance_id, duration_s in deselected_durations_s.items()
    if utterance_id in newly_selected
  )

  logger = getLogger(__name__)
  logger.info(
    f"Duration of utterances: {newly_selected_duration_s/60:.2f}min.")

  return newly_selected
