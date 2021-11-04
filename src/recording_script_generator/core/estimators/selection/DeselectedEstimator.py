from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


class DeselectedEstimator():
  def fit(self):
    pass

  def estimate(self, selection: Selection, utterances: Utterances) -> OrderedSet[UtteranceId]:
    deselected_utterance_ids = OrderedSet(utterances.keys()) - selection
    return deselected_utterance_ids
