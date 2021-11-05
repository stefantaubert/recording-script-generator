
from logging import getLogger
from typing import Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def add_to_selection_inplace(selection: Selection, add: OrderedSet[UtteranceId]) -> None:
  logger = getLogger(__name__)
  old_count = len(selection)
  selection |= add
  new_count = len(selection)
  added_count = new_count - old_count

  if added_count == 0:
    logger.info("Nothing was added to the selection.")
  else:
    logger.info(
        f"Added {added_count} utterances to the selection.")


def remove_from_selection_inplace(selection: Selection, remove: Set[UtteranceId]) -> None:
  logger = getLogger(__name__)
  old_count = len(selection)
  selection -= remove
  new_count = len(selection)
  removed_count = new_count - old_count

  if removed_count == 0:
    logger.info("Nothing was removed from the selection.")
  else:
    logger.info(
        f"Removed {removed_count} utterances from the selection.")


def select_selection_inplace(selection: Selection, select: OrderedSet[UtteranceId], utterances: Utterances) -> None:
  remove = selection - select
  add = select - selection
  add_to_selection_inplace(selection, add)
  remove_from_selection_inplace(selection, remove)
  assert selection == Selection(select)
