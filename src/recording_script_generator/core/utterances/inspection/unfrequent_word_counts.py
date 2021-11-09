from collections import Counter
from logging import getLogger
from typing import Dict, List, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances)


def get_minimum_frequency(words: List[str], word_frequencies: Counter) -> int:
  result = min(word_frequencies[word] for word in words)
  # for freq, word in zip(freqs, words):
  #   if freq <= 1:
  #     #print(word, freq)
  #     pass
  return result


def get_non_punctuation_words(utterance: Utterance) -> List[str]:
  assert isinstance(utterance, str)
  utterance = utterance.lower()
  words = utterance.split(" ")
  words_non_punctuation = strip_punctuation_words(words)
  return words_non_punctuation


def get_utterances_with_unfrequent_words(utterances: Utterances, min_occurrence_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting unfrequend words...")
  stripped_words: Dict[int, List[str]] = execute_method_on_utterances_mp(
    utterances=utterances,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    n_jobs=n_jobs,
    method=get_non_punctuation_words,
  )

  logger = getLogger("Getting counts...")
  words_counter = Counter(word for words in stripped_words.values()
                          for word in words)
  logger = getLogger("Done.")
  remove = OrderedSet()
  for utterance_id, words in stripped_words.items():
    min_freq = get_minimum_frequency(words, words_counter)

    if min_freq < min_occurrence_count:
      remove.add(utterance_id)

  return remove
