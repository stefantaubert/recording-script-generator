
from logging import getLogger
from typing import Dict, Set, Tuple

from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def remove_utterances_with_logging(utterance_ids: Set[UtteranceId], utterances: Utterances) -> None:
  logger = getLogger(__name__)
  if len(utterances) == 0 or len(utterance_ids) == 0:
    logger.info("Nothing to remove.")
    return

  assert len(utterances) > 0
  old_count = len(utterances)
  log_count = 10
  for utterance_id in list(utterance_ids)[:log_count]:
    utterance_str = ''.join(utterances[utterance_id])
    logger.info(f"Removing \"{utterance_str}\" ({utterance_id})...")

  if len(utterance_ids) > log_count:
    logger.info(f"Removing {len(utterance_ids) - log_count} further utterance(s)...")

  for utterance_id in utterance_ids:
    utterances.pop(utterance_id)

  new_count = len(utterances)

  logger.info(
      f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")


class RemoveTransformer():
  def fit(self):
    pass

  def transform(self, utterance_id: UtteranceId, remove: bool) -> Utterances:
    self.remove: Set[UtteranceId] = {
      utterance_id for utterance_id, dont_include in remove.items() if dont_include
    }
    # todo maybe check are valid utterance_ids

    remove_utterances_with_logging(self.remove, utterances)
    return utterances

  # def transform(self, utterances: Utterances, remove: Dict[UtteranceId, bool]) -> Utterances:
  #   self.remove: Set[UtteranceId] = {
  #     utterance_id for utterance_id, dont_include in remove.items() if dont_include
  #   }
  #   # todo maybe check are valid utterance_ids

  #   remove_utterances_with_logging(self.remove, utterances)
  #   return utterances
