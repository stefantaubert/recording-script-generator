from collections import Counter
from typing import Dict, List, Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.text_extraction import \
    strip_punctuation_words
from recording_script_generator.core.types import UtteranceId, Utterances


def get_minimum_frequency(words: List[str], word_frequencies: Counter) -> int:
  frequencies = [word_frequencies[word.lower()] for word in words]
  result = min(frequencies)
  # for freq, word in zip(freqs, words):
  #   if freq <= 1:
  #     #print(word, freq)
  #     pass
  return result


class UnfrequentWordCountEstimator():
  def fit(self, min_occurrence_count: Optional[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    self.min_occurrence_count = min_occurrence_count

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    stripped_words: Dict[int, List[str]] = {}
    for utterance_id, utterance_symbols in utterances.items():
      utterance = ''.join(utterance_symbols)
      words = utterance.split(" ")
      words_non_punctuation = strip_punctuation_words(words)
      stripped_words[utterance_id] = words_non_punctuation

    words_counter = Counter(word.lower() for words in stripped_words.values()
                            for word in words)
    remove = OrderedSet()
    for utterance_id, words in stripped_words.items():
      min_freq = get_minimum_frequency(words, words_counter)

      if min_freq < self.min_occurrence_count:
        remove.add(utterance_id)

    return remove
