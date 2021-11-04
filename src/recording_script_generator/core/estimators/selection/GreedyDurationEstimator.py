from logging import getLogger
from typing import Dict, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from text_selection import SelectionMode, greedy_ngrams_durations_advanced
from text_utils.types import Symbol


class GreedyDurationEstimator():
  def fit(self, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], mode: SelectionMode):
    self.n_gram = n_gram
    self.seconds = minutes * 60
    self.ignore_symbols = ignore_symbols
    self.mode = mode

  def estimate(self, utterances: Utterances, utterance_durations_s: Dict[UtteranceId, float]) -> OrderedSet[UtteranceId]:
    newly_selected = greedy_ngrams_durations_advanced(
      data=utterances,
      n_gram=self.n_gram,
      ignore_symbols=self.ignore_symbols,
      target_duration=self.seconds,
      durations=utterance_durations_s,
      mode=self.mode,
    )

    newly_selected_duration_s = sum(
      duration_s
      for utterance_id, duration_s in utterance_durations_s.items()
      if utterance_id in newly_selected
    )

    logger = getLogger(__name__)
    logger.info(
      f"Duration of utterances: {newly_selected_duration_s/60:.2f}min.")

    return newly_selected