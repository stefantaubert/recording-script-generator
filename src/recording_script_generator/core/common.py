from enum import IntEnum
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations,
                                                   UtteranceId, Utterances,
                                                   utterance_to_text)


def log_utterances(utterances: Utterances, selection: Set[UtteranceId], log_count: Optional[int] = 10) -> None:
  logger = getLogger(__name__)
  if log_count is None:
    log_count = len(selection)
  logger.info("Utterances:")
  for utterance_id in list(sorted(selection))[:log_count]:
    utterance_str = utterance_to_text(utterances[utterance_id])
    logger.info(f"- \"{utterance_str}\" ({utterance_id}),")
  if log_count < len(selection):
    logger.info(f"- and {len(selection) - log_count} further utterance(s)...")
