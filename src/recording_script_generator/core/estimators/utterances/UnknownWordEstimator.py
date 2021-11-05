from functools import partial
from logging import getLogger
from typing import List, Optional, Set

import enchant
from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import (
    UtteranceEstimatorBase, execute_method_on_utterances_mp_bool)
from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import UtteranceId, Utterances
from text_utils.language import Language
from text_utils.types import Symbols


def get_non_dict_words_amount(words: List[str], lexicon: enchant.Dict) -> int:
  tmp = []
  for word in words:
    assert word != ""
    val = int(lexicon.check(word))
    tmp.append(val)
  words_in_dict = sum(tmp)
  words_not_in_dict = len(tmp) - words_in_dict
  return words_not_in_dict


def main(symbols: Symbols, max_unknown_word_count: int, lexicon: enchant.Dict) -> bool:
  symbols = ''.join(symbols)
  words = symbols.split(" ")
  words_non_punctuation = strip_punctuation_words(words)

  non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, lexicon)
  if non_dict_words_amount > max_unknown_word_count:
    return True

  return False


def get_utterances_with_non_dictionary_words(utterances: Utterances, max_unknown_word_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting non-dictionary words...")

  if utterances.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  lexicon = enchant.Dict("en_US")

  method = partial(
    main,
    lexicon=lexicon,
    max_unknown_word_count=max_unknown_word_count,
  )

  return execute_method_on_utterances_mp_bool(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  )


class UnknownWordEstimator(UtteranceEstimatorBase):
  def fit(self, max_unknown_word_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    super().fit(n_jobs, maxtasksperchild, chunksize)
    self.lexicon = enchant.Dict("en_US")
    self.max_unknown_word_count = max_unknown_word_count

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    logger = getLogger(__name__)
    logger.info("Detecting non-dictionary words...")

    if utterances.language != Language.ENG:
      logger = getLogger(__name__)
      logger.error("Language needs to be English!")
      raise Exception()

    method = partial(
      main,
      lexicon=self.lexicon,
      max_unknown_word_count=self.max_unknown_word_count,
    )

    return super().estimate(
      utterances=utterances,
      method=method,
    )
