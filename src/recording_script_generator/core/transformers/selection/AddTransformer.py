from logging import getLogger

from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def get_selection_added(selection: Selection, add: OrderedSet[UtteranceId], utterances: Utterances) -> Selection:
  logger = getLogger(__name__)
  result = Selection(selection | add)
  if len(add) == 0:
    logger.info("Nothing was added to the selection.")
  else:
    logger.info("Selected:")
    for utterance_id in add:
      logger.info(f"- {utterance_id}: {''.join(utterances[utterance_id])}")

    #old_count = len(selection)
    #new_count = len(result)

    # logger.info(
    #     f"Selected {new_count-old_count} ({(new_count- old_count)/new_count*100:.2f}%) and obtained a selection of {new_count} utterances.")

  return result


class AddTransformer():
  def fit(self):
    pass

  def transform(self, selection: Selection, add: OrderedSet[UtteranceId], utterances: Utterances) -> Selection:
    logger = getLogger(__name__)
    if len(add) == 0:
      logger.info("Nothing to add to selection.")
      return

    old_count = len(selection)

    result = Selection(selection | add)

    new_count = len(result)

    logger.info("Selected:")
    for utterance_id in add:
      logger.info(f"- {utterance_id}: {''.join(utterances[utterance_id])}")

    # logger.info(
    #     f"Selected {new_count-old_count} ({(new_count- old_count)/new_count*100:.2f}%) and obtained a selection of {new_count} utterances.")

    return result
