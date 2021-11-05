from functools import partial
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import UtteranceId, Utterances
from text_utils.types import Symbols


def contains_undesired_text(sentence: str, undesired: Set[str], ignore_case: bool) -> bool:
  for x in undesired:
    if ignore_case:
      if x.lower() in sentence.lower():
        return True
    else:
      if x in sentence:
        return True

  return False


def main(symbols: Symbols, min_count: Optional[int], max_count: Optional[int]) -> bool:
  symbols = ''.join(symbols)
  words = symbols.split(" ")
  words_count = len(words)

  if min_count is not None and words_count < min_count:
    return True
  if max_count is not None and words_count > max_count:
    return True

  return False


def get_utterances_with_custom_word_counts(utterances: Utterances, min_count: Optional[int], max_count: Optional[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[UtteranceId]:
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
  )
