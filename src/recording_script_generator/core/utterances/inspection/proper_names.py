import re
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.helper import strip_punctuation_utterance, strip_punctuation_words
from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances,
                                                   utterance_to_str)
from text_utils.language import Language

pattern = re.compile(r" [A-HJ-Z]")


def contains_eng_proper_names(utterance_str: str) -> bool:
  matches = re.search(pattern, utterance_str)
  return matches is not None


def main(utterance: Utterance) -> bool:
  utterance_str = utterance_to_str(utterance)
  stripped_utterance = strip_punctuation_utterance(utterance_str)
  result = contains_eng_proper_names(stripped_utterance)
  return result


def get_utterances_with_proper_names(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting proper names...")
  if utterances.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  return execute_method_on_utterances_mp_bool(
    utterances=utterances,
    method=main,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
