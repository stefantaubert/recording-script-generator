from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances


class AllEstimator():
  def fit(self):
    pass

  def estimate(self, utterances: Utterances) -> OrderedSet[UtteranceId]:
    result = OrderedSet(utterances.keys())
    return result
