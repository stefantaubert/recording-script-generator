from functools import partial
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances,
                                                   utterance_to_text)


def contains_undesired_text(utterance: str, undesired: Set[str], ignore_case: bool) -> bool:
  for x in undesired:
    if ignore_case:
      if x.lower() in utterance.lower():
        return True
    else:
      if x in utterance:
        return True

  return False


def main(utterance: Utterance, undesired: Set[str]) -> bool:
  utterance_str = utterance_to_text(utterance)
  result = contains_undesired_text(utterance_str, undesired=undesired, ignore_case=True)
  return result


def get_utterances_with_undesired_text(utterances: Utterances, undesired: Set[str], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting undesired words/symbols...")

  method = partial(
    main,
    undesired=undesired,
  )

  return execute_method_on_utterances_mp_bool(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
