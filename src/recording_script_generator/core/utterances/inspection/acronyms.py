from functools import partial
from logging import getLogger
from typing import List, Optional, Set

from recording_script_generator.core.helper import strip_punctuation_words
from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances,
                                                   utterance_to_text)


def words_contain_acronyms(words: List[str], min_acronym_len: int) -> bool:
  return any(is_acronym(word, min_acronym_len) for word in words)


def is_acronym(word: str, min_acronym_len: int) -> bool:
  if len(word) >= min_acronym_len and word.isupper():
    return True
  return False


def main(utterance: Utterance, min_acronym_len: int) -> bool:
  utterance_str = utterance_to_text(utterance)
  words = utterance_str.split(" ")
  words_non_punctuation = strip_punctuation_words(words)
  result = words_contain_acronyms(words_non_punctuation, min_acronym_len)
  return result


def get_utterances_with_acronyms(utterances: Utterances, min_acronym_len: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting acronyms...")

  method = partial(
    main,
    min_acronym_len=min_acronym_len,
  )

  return execute_method_on_utterances_mp_bool(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
