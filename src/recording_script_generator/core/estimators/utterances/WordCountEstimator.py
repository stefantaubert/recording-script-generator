from functools import partial
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


def main(symbols: Symbols, min_count: Optional[int], max_count: Optional[int]) -> bool:
  symbols = ''.join(symbols)
  words = symbols.split(" ")
  words_count = len(words)

  if min_count is not None and words_count < min_count:
    return True
  if max_count is not None and words_count > max_count:
    return True

  return False


class WordCountEstimator(UtteranceEstimatorBase):
  def fit(self, min_count: Optional[int], max_count: Optional[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    super().fit(n_jobs, maxtasksperchild, chunksize)
    self.min_count = min_count
    self.max_count = max_count

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    method = partial(
      main,
      min_count=self.min_count,
      max_count=self.max_count,
    )

    return super().estimate(
      utterances=utterances,
      method=method,
    )
