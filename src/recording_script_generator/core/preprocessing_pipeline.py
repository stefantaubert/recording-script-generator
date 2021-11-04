from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from multiprocessing.pool import RemoteTraceback
from multiprocessing.sharedctypes import RawValue
from sys import getsizeof
from time import perf_counter, sleep
from typing import Any, Dict, List, Tuple, cast

from recording_script_generator.core.estimators.AcronymEstimator import \
    AcronymEstimator
from recording_script_generator.core.transformers.ChunkingTransformer import \
    ChunkingTransformer
from recording_script_generator.core.transformers.DechunkingTransformer import \
    DechunkingTransformer
from recording_script_generator.core.transformers.RemoveSelectionTransformer import \
    RemoveSelectionTransformer
from recording_script_generator.core.transformers.RemoveTransformer import \
    RemoveTransformer
from recording_script_generator.core.types import (Selection, Utterance,
                                                   UtteranceId, Utterances)
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import SymbolId, Symbols
from tqdm import tqdm

utterances: Dict[UtteranceId, Symbols]


def chunk_pipeline(utterance_id: UtteranceId):
  # logger = getLogger(__name__)
  global utterances
  assert utterance_id in utterances
  symbols = utterances[utterance_id]
  utterance = utterance_id, symbols
  acronym_estimator = AcronymEstimator()
  remove = acronym_estimator.estimate(utterance)
  utterance_id, _ = utterance
  return utterance_id, remove


def handle_error(exception: Exception) -> None:
  logger = getLogger(__name__)
  #tb = sys.exc_info()
  # traceback.print_stack()
  # print(traceback.format_exc())
  logger.exception(exception)
  remote_traceback = cast(RemoteTraceback, exception.__cause__)
  logger.info(remote_traceback.tb)
  pass


def init_pool(utts: Utterances):
  global utterances
  utterances = utts


def do_pipeline(utterances: Utterances, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: int):
  logger = getLogger(__name__)
  acronym_estimator = AcronymEstimator()
  remove_transformer = RemoveTransformer()
  chunking_transformer = ChunkingTransformer()
  dechunking_tranformer = DechunkingTransformer()
  remove_selection_transformer = RemoveSelectionTransformer()

  chunking_transformer.fit()
  remove_transformer.fit()
  dechunking_tranformer.fit()
  remove_selection_transformer.fit()

  logger.info(f"Size of utterances in memory: {getsizeof(utterances)/1024**3:.2f} Gb")

  acronym_estimator.fit(
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  )

  result = acronym_estimator.estimate(utterances)

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
  utterances = remove_transformer.transform(utterances, result)
  selection = remove_selection_transformer.transform(selection, utterances)

  return utterances, selection
