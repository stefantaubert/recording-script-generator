from copy import deepcopy
from logging import getLogger
from typing import Set

from recording_script_generator.core.types import UtteranceId, Utterances
from tqdm import tqdm


def get_utterances_removed(utterances: Utterances, remove: Set[UtteranceId]) -> Utterances:
  # todo maybe check are valid utterance_ids
  logger = getLogger(__name__)
  logger.info("Removing utterances...")
  result = Utterances({k: v for k, v in tqdm(utterances.items()) if k not in remove})
  result.language = utterances.language
  result.symbol_format = utterances.symbol_format

  if len(remove) == 0:
    logger.info("Nothing was removed.")
  else:
    log_count = 10
    for utterance_id in list(sorted(remove))[:log_count]:
      utterance_str = ''.join(utterances[utterance_id])
      logger.info(f"Removed \"{utterance_str}\" ({utterance_id})")
    if len(remove) > log_count:
      logger.info(f"Removed {len(remove) - log_count} further utterance(s).")

    old_count = len(utterances)
    new_count = len(result)
    logger.info(
        f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")

  return result


class RemoveTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances, remove: Set[UtteranceId]) -> Utterances:
    # todo maybe check are valid utterance_ids
    logger = getLogger(__name__)
    result = deepcopy(utterances)
    if len(utterances) == 0 or len(remove) == 0:
      logger.info("Nothing to remove.")
      return result

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
