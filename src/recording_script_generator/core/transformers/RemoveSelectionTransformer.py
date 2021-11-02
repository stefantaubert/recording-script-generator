from logging import getLogger
from typing import Dict, Set, Tuple

from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def remove_utterances_from_selection_with_logging(utterance_ids: Set[UtteranceId], selection: Selection) -> None:
  logger = getLogger(__name__)
  if len(selection) == 0 or len(utterance_ids) == 0:
    logger.info("Nothing to deselect.")
    return

  old_count = len(selection)

  selection -= utterance_ids

  new_count = len(selection)

  logger.info(
      f"Deselected {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained a selection of {new_count} utterances.")


class RemoveSelectionTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances, selection: Selection) -> Selection:
    utterances_not_present_anymore = selection - set(utterances.keys())
    remove_utterances_from_selection_with_logging(utterances_not_present_anymore, selection)
    return selection
