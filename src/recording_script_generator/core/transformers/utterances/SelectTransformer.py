from copy import deepcopy
from logging import getLogger
from typing import Set

from recording_script_generator.core.types import UtteranceId, Utterances


class SelectTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances, select: Set[UtteranceId]) -> Utterances:
    # todo maybe check are valid utterance_ids
    logger = getLogger(__name__)
    remove = utterances.keys() - keys
    for key in remove:
      utterances.pop(key)

    return result


def sync_utterances(utterances: Utterances, keys: Set[UtteranceId]) -> None:
  remove = utterances.keys() - keys
  for key in remove:
    utterances.pop(key)
