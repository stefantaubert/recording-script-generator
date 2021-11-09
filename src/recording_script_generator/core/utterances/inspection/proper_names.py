import re
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances)
from text_utils.language import Language

pattern = re.compile(r" [A-HJ-Z]")


def contains_eng_proper_names(utterance_str: str) -> bool:
  matches = re.search(pattern, utterance_str)
  return matches is not None


def main(utterance: Utterance) -> bool:
  assert isinstance(utterance, str)
  result = contains_eng_proper_names(utterance)
  return result


def get_utterances_with_proper_names(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[UtteranceId]:
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
  )
