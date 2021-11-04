

from concurrent.futures.process import ProcessPoolExecutor
from logging import getLogger
from multiprocessing import Pool
from time import perf_counter
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


utterances_shared_memory: Utterances


def main(utterance_id: UtteranceId):
  # pylint: disable=global-variable-not-assigned
  global utterances_shared_memory
  symbols = utterances_shared_memory[utterance_id]
  symbols = ''.join(symbols)
  words = symbols.split(" ")
  words_non_punctuation = strip_punctuation_words(words)

  result = words_contain_acronyms(words_non_punctuation)
  return utterance_id, result


def init_pool(utterances: Utterances):
  global utterances_shared_memory
  utterances_shared_memory = utterances


class AcronymEstimator():
  def fit(self, n_jobs: int, maxtasksperchild: int, chunksize: int):
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    #self.utterances = utterances

  def estimate(self, utterances: Utterances) -> Dict[UtteranceId, bool]:
    start = perf_counter()
    with Pool(
        processes=self.n_jobs,
        initializer=init_pool,
        initargs=(utterances,),
        maxtasksperchild=self.maxtasksperchild,
      ) as pool:
      transformed_utterances: Dict[UtteranceId, Symbols] = dict(tqdm(
        pool.imap_unordered(main, utterances.keys(), chunksize=self.chunksize),
        total=len(utterances),
      ))
    end = perf_counter()
    logger = getLogger(__name__)
    logger.info(f"Duration: {end-start:.2f}s")

    return transformed_utterances
