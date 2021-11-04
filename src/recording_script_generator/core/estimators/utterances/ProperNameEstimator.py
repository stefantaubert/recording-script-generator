import re
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    UtteranceEstimatorBase
from recording_script_generator.core.types import UtteranceId, Utterances
from text_utils.language import Language
from text_utils.types import Symbols

pattern = re.compile(r" [A-HJ-Z]")


def contains_eng_proper_names(sentence: str) -> bool:
  matches = re.search(pattern, sentence)
  return matches is not None


def main(symbols: Symbols) -> bool:
  symbols = ''.join(symbols)
  result = contains_eng_proper_names(symbols)
  return result


class ProperNameEstimator(UtteranceEstimatorBase):
  def fit(self, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    super().fit(n_jobs, maxtasksperchild, chunksize)

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    logger = getLogger(__name__)
    logger.info("Detecting proper names...")
    if utterances.language != Language.ENG:
      logger = getLogger(__name__)
      logger.error("Language needs to be English!")
      raise Exception()

    return super().estimate(
      utterances=utterances,
      method=main,
    )
