from functools import partial
from logging import getLogger
from typing import List, Optional, Set

import enchant
from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp_bool
from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances)
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


def main(utterance: Utterance, max_unknown_word_count: int, lexicon: enchant.Dict) -> bool:
  assert isinstance(utterance, str)
  words = utterance.split(" ")
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
