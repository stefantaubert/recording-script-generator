from logging import getLogger
from typing import Set

from recording_script_generator.core.types import UtteranceId, Utterances


class RemoveUtterancesTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances, remove: Set[UtteranceId]) -> Utterances:
    # todo maybe check are valid utterance_ids
    logger = getLogger(__name__)
    if len(utterances) == 0 or len(remove) == 0:
      logger.info("Nothing to remove.")
      return

    result = utterances.copy()

    assert len(result) > 0
    old_count = len(result)
    log_count = 10
    for utterance_id in list(remove)[:log_count]:
      utterance_str = ''.join(result[utterance_id])
      logger.info(f"Removing \"{utterance_str}\" ({utterance_id})...")

    if len(remove) > log_count:
      logger.info(f"Removing {len(remove) - log_count} further utterance(s)...")

    for utterance_id in remove:
      result.pop(utterance_id)

    new_count = len(result)

    logger.info(
        f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")
    return result
