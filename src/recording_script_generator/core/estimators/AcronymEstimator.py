

from concurrent.futures.process import ProcessPoolExecutor
from typing import Dict, List, Tuple

from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances)
from text_utils.types import Symbols
from tqdm import tqdm


def words_contain_acronyms(words: List[str]) -> bool:
  return any(is_acronym(word) for word in words)


def is_acronym(word: str) -> bool:
  if len(word) >= 3 and word.isupper():
    return True
  return False


class AcronymEstimator():
  def fit(self):
    pass
    #self.utterances = utterances

  def estimate(self, utterance: Utterance) -> bool:
    _, symbols = utterance
    symbols = ''.join(symbols)
    words = symbols.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    result = words_contain_acronyms(words_non_punctuation)
    return result
