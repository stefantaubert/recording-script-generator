from logging import getLogger
from typing import Set

from recording_script_generator.core.types import UtteranceId, Utterances
from tqdm import tqdm


def remove_from_utterances_inplace(utterances: Utterances, remove: Set[UtteranceId]) -> None:
  # todo maybe check are valid utterance_ids
  logger = getLogger(__name__)
  logger.info("Removing utterances...")
  old_count = len(utterances)
  if len(remove) == 0:
    logger.info("Nothing was removed.")
  else:
    for k in remove:
      utterances.pop(k)

    new_count = len(utterances)
    logger.info(
        f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")


def add_to_utterances_inplace(utterances: Utterances, add: Set[UtteranceId], add_from: Utterances) -> None:
  # todo maybe check are valid utterance_ids
  logger = getLogger(__name__)
  logger.info("Adding utterances...")
  old_count = len(utterances)
  if len(add) == 0:
    logger.info("Nothing to add.")
  else:
    for k in add:
      assert k in add_from
      assert k not in utterances
      utterances[k] = add_from[k]
    new_count = len(utterances)
    logger.info(f"Added {new_count- old_count} utterances and obtained {new_count} utterances.")
    # logger.info(
    #     f"Added {new_count- old_count} to {old_count} utterances ({(new_count- old_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")
