from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


class RemovedEstimator():
  def fit(self):
    pass

  def estimate(self, selection: Selection, utterances: Utterances) -> OrderedSet[UtteranceId]:
    utterances_not_present_anymore = selection - set(utterances.keys())
    return utterances_not_present_anymore
