from functools import partial
from logging import getLogger
from multiprocessing import Pool
from time import perf_counter
from typing import Callable, Dict, Optional, Set, Tuple, TypeVar

from recording_script_generator.core.types import (Utterance, UtteranceId,
                                                   Utterances)
from tqdm import tqdm

utterances_shared_memory: Utterances


def init_pool(utterances: Utterances):
  global utterances_shared_memory
  utterances_shared_memory = utterances


def main_proxy(utterance_id: UtteranceId, main_method: Callable[[Utterance], bool]) -> Tuple[UtteranceId, bool]:
  # pylint: disable=global-variable-not-assigned
  global utterances_shared_memory
  utterance = utterances_shared_memory[utterance_id]
  result_method = main_method(utterance)
  return utterance_id, result_method


T = TypeVar('T')


def execute_method_on_utterances_mp(utterances: Utterances, method: Callable[[Utterance], T], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Dict[UtteranceId, T]:
  start = perf_counter()

  method_proxy = partial(
    main_proxy,
    main_method=method,
  )

  with Pool(
    processes=n_jobs,
    initializer=init_pool,
    initargs=(utterances,),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    transformed_utterances: Dict[UtteranceId, T] = dict(tqdm(
      pool.imap_unordered(method_proxy, utterances.keys(), chunksize=chunksize),
      total=len(utterances),
    ))

  end = perf_counter()
  logger = getLogger(__name__)
  logger.info(f"Duration: {end-start:.2f}s")
  return transformed_utterances


def execute_method_on_utterances_mp_bool(utterances: Utterances, method: Callable[[Utterance], bool], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[UtteranceId]:
  transformed_utterances = execute_method_on_utterances_mp(
    utterances=utterances,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    method=method,
    n_jobs=n_jobs,
  )

  remove: Set[UtteranceId] = {
    utterance_id for utterance_id, dont_include in transformed_utterances.items() if dont_include
  }

  return remove

  # with Pool(
  #     processes=n_jobs,
  #     initializer=init_pool,
  #     initargs=(utterances,),
  #     maxtasksperchild=maxtasksperchild,
  #   ) as pool:
  #   start = perf_counter()
  #   transformed_utterances: Dict[UtteranceId, Symbols] = dict(tqdm(
  #     pool.imap_unordered(, utterances.keys(), chunksize=chunksize),
  #     total=len(utterances),
  #   ))
  #   # transformed_utterances: Dict[UtteranceId, Symbols] = dict()
  #   # with tqdm(total=len(keys)) as pbar:
  #   #   iterator = pool.imap_unordered(method, keys, chunksize=chunksize_inner)
  #   #   for utterance_id, include in iterator:
  #   #     transformed_utterances[utterance_id] = include
  #   #     pbar.update()
  #   logger.info(f"Duration: {perf_counter() - start:.2f}s")

  # not currently usable: https://github.com/python/cpython/pull/27373
  # with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_pool, initargs=(utterances,)) as ex:
  #   start = perf_counter()
  #   transformed_utterances: Dict[UtteranceId, Symbols] = dict(tqdm(
  #     ex.map(method, keys, chunksize=chunksize_inner),
  #     total=len(keys),
  #   ))
  #   logger.info(f"Duration: {perf_counter() - start:.2f}s")
