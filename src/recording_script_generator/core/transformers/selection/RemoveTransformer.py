from logging import getLogger
from typing import Set

from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


class RemoveTransformer():
  def fit(self):
    pass

  def transform(self, selection: Selection, remove: Set[UtteranceId], utterances: Utterances) -> Selection:
    logger = getLogger(__name__)
    if len(selection) == 0:
      logger.info("Nothing to deselect from.")
      return

    if len(remove) == 0:
      logger.info("Nothing to deselect.")
      return

    old_count = len(selection)

    result = Selection(selection - remove)

    new_count = len(result)

    logger.info("Deselected:")
    for utterance_id in remove:
      logger.info(f"- {utterance_id}: {''.join(utterances[utterance_id])}")

    logger.info(
        f"Deselected {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained a selection of {new_count} utterances.")

    return result
