from typing import List, Optional, Set

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    UtteranceEstimatorBase
from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import UtteranceId, Utterances
from text_utils.types import Symbols


def words_contain_acronyms(words: List[str]) -> bool:
  return any(is_acronym(word) for word in words)


def is_acronym(word: str) -> bool:
  if len(word) >= 3 and word.isupper():
    return True
  return False


def main(symbols: Symbols) -> bool:
  symbols = ''.join(symbols)
  words = symbols.split(" ")
  words_non_punctuation = strip_punctuation_words(words)

  result = words_contain_acronyms(words_non_punctuation)
  return result


class AcronymEstimator(UtteranceEstimatorBase):
  def fit(self, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    super().fit(n_jobs, maxtasksperchild, chunksize)

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    return super().estimate(
      utterances=utterances,
      method=main,
    )
