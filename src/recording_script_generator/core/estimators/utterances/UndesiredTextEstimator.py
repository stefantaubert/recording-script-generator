from functools import partial
from logging import getLogger
from typing import Optional, Set

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    UtteranceEstimatorBase
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


def main(symbols: Symbols, undesired: Set[str]) -> bool:
  symbols = ''.join(symbols)
  result = contains_undesired_text(symbols, undesired=undesired, ignore_case=True)
  return result


class UndesiredTextEstimator(UtteranceEstimatorBase):
  def fit(self, undesired: Set[str], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    super().fit(n_jobs, maxtasksperchild, chunksize)
    self.undesired = undesired

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    logger = getLogger(__name__)
    logger.info("Detecting undesired words/symbols...")
    method = partial(
      main,
      undesired=self.undesired,
    )

    return super().estimate(
      utterances=utterances,
      method=method,
    )
