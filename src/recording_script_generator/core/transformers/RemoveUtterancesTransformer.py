
from copy import deepcopy
from logging import getLogger
from typing import Dict, Set, Tuple

from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def remove_utterances_with_logging(utterance_ids: Set[UtteranceId], utterances: Utterances) -> Utterances:
  logger = getLogger(__name__)
  if len(utterances) == 0 or len(utterance_ids) == 0:
    logger.info("Nothing to remove.")
    return

  result = utterances.copy()

  assert len(result) > 0
  old_count = len(result)
  log_count = 10
  for utterance_id in list(utterance_ids)[:log_count]:
    utterance_str = ''.join(result[utterance_id])
    logger.info(f"Removing \"{utterance_str}\" ({utterance_id})...")

  if len(utterance_ids) > log_count:
    logger.info(f"Removing {len(utterance_ids) - log_count} further utterance(s)...")

  for utterance_id in utterance_ids:
    result.pop(utterance_id)

  new_count = len(result)

  logger.info(
      f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")
  return result


class RemoveUtterancesTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances, remove: Dict[UtteranceId, bool]) -> Utterances:
    remove: Set[UtteranceId] = {
      utterance_id for utterance_id, dont_include in remove.items() if dont_include
    }

    # todo maybe check are valid utterance_ids
    utterances = remove_utterances_with_logging(remove, utterances)
    return utterances

  # def transform(self, utterances: Utterances, remove: Dict[UtteranceId, bool]) -> Utterances:
  #   self.remove: Set[UtteranceId] = {
  #     utterance_id for utterance_id, dont_include in remove.items() if dont_include
  #   }
  #   # todo maybe check are valid utterance_ids

  #   remove_utterances_with_logging(self.remove, utterances)
  #   return utterances
