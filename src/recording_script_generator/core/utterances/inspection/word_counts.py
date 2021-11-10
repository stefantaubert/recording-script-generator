from functools import partial
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances,
                                                   utterance_to_str)


def main(utterance: Utterance, min_count: Optional[int], max_count: Optional[int]) -> bool:
  utterance_str = utterance_to_str(utterance)
  words = utterance_str.split(" ")
  words_count = len(words)

  if min_count is not None and words_count < min_count:
    return True
  if max_count is not None and words_count > max_count:
    return True

  return False


def get_utterances_with_custom_word_counts(utterances: Utterances, min_count: Optional[int], max_count: Optional[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting words counts...")
  method = partial(
    main,
    min_count=min_count,
    max_count=max_count,
  )

  return execute_method_on_utterances_mp_bool(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
