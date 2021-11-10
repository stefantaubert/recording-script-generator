import gc
import math
from collections import Counter
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

from ordered_set import OrderedSet
from recording_script_generator.core.helper import strip_punctuation_utterance
from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances,
                                                   utterance_to_str)
from tqdm import tqdm


def get_minimum_frequency(words: List[str], word_frequencies: Counter) -> int:
  result = min(word_frequencies[word] for word in words)
  # for freq, word in zip(freqs, words):
  #   if freq <= 1:
  #     #print(word, freq)
  #     pass
  assert result > 0
  return result


def main_step_1(utterance: Utterance) -> str:
  utterance_str = utterance_to_str(utterance)
  utterance_str = utterance_str.lower()
  stripped_utterance = strip_punctuation_utterance(utterance_str)
  return stripped_utterance


process_stripped_utterances: Dict[UtteranceId, str] = None
process_counter: Counter = None


def init_pool_step_2(stripped_utterances: Dict[UtteranceId, str], counter: Counter):
  global process_stripped_utterances
  global process_counter
  process_stripped_utterances = stripped_utterances
  process_counter = counter


def main_step_2(utterance_id: UtteranceId, min_occurrence_count: int) -> Tuple[UtteranceId, bool]:
  global process_stripped_utterances
  global process_counter
  utterance = process_stripped_utterances[utterance_id]
  words = utterance.split(" ")
  min_freq = get_minimum_frequency(words, process_counter)
  result = min_freq < min_occurrence_count
  return utterance_id, result


def get_utterances_with_unfrequent_words(utterances: Utterances, min_occurrence_count: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting unfrequent words...")

  stripped_utterances = execute_method_on_utterances_mp(
    utterances=utterances,
    method=main_step_1,
    batches=batches,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
  )

  if batches is None:
    assert chunksize is not None
  else:
    chunksize = math.ceil(len(utterances) / n_jobs / batches)

  logger.info("Converting to str...")
  words = [
    word
    for utterance in tqdm(stripped_utterances.values())
    for word in utterance.split(" ")
  ]
  #total_string = ' '.join(stripped_utterances.values())
  #logger.info("Converting to words...")
  # words = total_string.split(" ")
  # del total_string
  logger.info("Converting to counter...")
  total_counter = Counter(words)
  del words
  gc.collect()
  logger.info("Done.")

  logger.info("Next step...")

  method = partial(
    main_step_2,
    min_occurrence_count=min_occurrence_count,
  )

  remove: Set[UtteranceId] = set()

  with Pool(
    processes=n_jobs,
    initializer=init_pool_step_2,
    initargs=(stripped_utterances, total_counter,),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    with tqdm(total=len(stripped_utterances)) as pbar:
      iterator = pool.imap_unordered(method, stripped_utterances.keys(), chunksize=chunksize)
      for utterance_id, dont_include in iterator:
        if dont_include:
          remove.add(utterance_id)
        pbar.update()
  del pool
  del stripped_utterances
  del total_counter
  gc.collect()

  return remove
