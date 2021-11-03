import itertools
import multiprocessing
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import Manager, Pool, managers, synchronize
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


def chunk_pipeline_v2(utterance_id: UtteranceId, utterances: Utterances, language: Language, symbol_format: SymbolFormat):
  logger = getLogger(__name__)
  assert utterance_id in utterances
  symbols = utterances[utterance_id]
  utterance = utterance_id, symbols
  #logger.info(f"start: {utterance_id}")
  #utterances = Utterances(zip(*chunk))
  #utterances.language = language
  #utterances.symbol_format = symbol_format

  #logger.info("restored input")
  # sleep(3)

  acronym_estimator = AcronymEstimator()
  remove_transformer = RemoveTransformer()
  remove = False

  remove = acronym_estimator.estimate(utterance)

  #logger.info(f"end: {utterance_id}")
  utterance_id, _ = utterance
  return utterance_id, remove


utterances: Dict[UtteranceId, Symbols]


def chunk_pipeline_v3(utterance_id: UtteranceId, language: Language, symbol_format: SymbolFormat):
  # logger = getLogger(__name__)
  global utterances
  assert utterance_id in utterances
  symbols = utterances[utterance_id]
  utterance = utterance_id, symbols
  #logger.info(f"start: {utterance_id}")
  #utterances = Utterances(zip(*chunk))
  #utterances.language = language
  #utterances.symbol_format = symbol_format

  #logger.info("restored input")
  # sleep(3)

  acronym_estimator = AcronymEstimator()
  #remove_transformer = RemoveTransformer()
  remove = False

  remove = acronym_estimator.estimate(utterance)
  del acronym_estimator
  del symbols

  #logger.info(f"end: {utterance_id}")
  utterance_id, _ = utterance
  return utterance_id, remove


def chunk_pipeline(utterance: Utterance, language: Language, symbol_format: SymbolFormat):
  logger = getLogger(__name__)
  logger.info("start")

  #utterances = Utterances(zip(*chunk))
  #utterances.language = language
  #utterances.symbol_format = symbol_format

  #logger.info("restored input")
  # sleep(3)

  acronym_estimator = AcronymEstimator()
  remove_transformer = RemoveTransformer()
  remove = False

  remove = acronym_estimator.estimate(utterance)

  # logger.info("stop")
  utterance_id, _ = utterance
  return utterance_id, remove


def _get_chunks(*iterables, chunksize):
  """ Iterates over zip()ed iterables in chunks. """
  it = zip(*iterables)
  while True:
    chunk = tuple(itertools.islice(it, chunksize))
    if not chunk:
      return
    yield chunk


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
  # print(utts[0])
  pass


def do_pipeline_v2(utterances: Dict[UtteranceId, Symbols], selection: Selection, language: Language, symbol_format: SymbolFormat, n_jobs: int, chunksize_inner: int):
  logger = getLogger(__name__)
  acronym_estimator = AcronymEstimator()
  remove_transformer = RemoveTransformer()
  chunking_transformer = ChunkingTransformer()
  dechunking_tranformer = DechunkingTransformer()
  remove_selection_transformer = RemoveSelectionTransformer()

  chunking_transformer.fit()
  acronym_estimator.fit()
  remove_transformer.fit()
  dechunking_tranformer.fit()
  remove_selection_transformer.fit()

  logger.info(f"Size of utterances: {getsizeof(utterances)/1024**3:.2f} Gb")

  method = partial(
    chunk_pipeline_v3,
    language=language,
    symbol_format=symbol_format
  )

  keys = utterances.keys()

  transformed_utterances = dict()

  with Pool(
      processes=n_jobs,
      initializer=init_pool,
      initargs=(utterances,),
      maxtasksperchild=1,
    ) as pool:
    start = perf_counter()
    # transformed_utterances: List[Any] = dict(tqdm(
    #   pool.imap_unordered(method, keys, chunksize=chunksize_inner),
    #   total=len(keys),
    # ))

    with tqdm(total=len(keys)) as pbar:
      iterator = pool.imap_unordered(method, keys, chunksize=chunksize_inner)
      for utterance_id, include in iterator:
        transformed_utterances[utterance_id] = include
        pbar.update()
    logger.info(f"Duration: {perf_counter() - start:.2f}s")

  with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_pool, initargs=(utterances,)) as ex:
    start = perf_counter()
    transformed_utterances: Dict[UtteranceId, Symbols] = dict(tqdm(
      ex.map(method, keys, chunksize=chunksize_inner),
      total=len(keys),
    ))
    logger.info(f"Duration: {perf_counter() - start:.2f}s")

  with Manager() as manager:
    manager = cast(SyncManager, manager)
    # ignore order
    #x = manager.list(list(utterances.items()))
    logger.info("Preparing multiprocessing...")
    start = perf_counter()
    #d = manager.dict({k: v for k, v in utterances.items()})
    d = manager.dict(utterances)

    logger.info(f"Duration: {perf_counter() - start:.2f}s")
    # manager.Array()
    #del utterances

    method = partial(
      chunk_pipeline_v2,
      utterances=d,
      language=language,
      symbol_format=symbol_format
    )

    # utterance_ids = list(d.keys())

    transformed_utterances = list()

    with Pool(processes=n_jobs, initializer=init_pool, initargs=(utterances,)) as pool:
      start = perf_counter()

      transformed_utterances: List[Any] = list(tqdm(
        pool.imap_unordered(method, d.keys(), chunksize=chunksize_inner),
        total=len(d),
      ))
      logger.info(f"Duration: {perf_counter() - start:.2f}s")

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
      start = perf_counter()
      transformed_utterances: List[Any] = list(tqdm(
        ex.map(method, d.keys(), chunksize=chunksize_inner),
        total=len(d)
      ))
      logger.info(f"Duration: {perf_counter() - start:.2f}s")

      # map_result = pool.map_async(
      #   func=method,
      #   iterable=d.keys(),
      #   chunksize=chunksize_inner,
      #     error_callback=handle_error
      # )
      # logger.info("All started...")
      # map_result.wait()
      # transformed_utterances = map_result.get()

      # logger.info(transformed_utterances)
      # with tqdm(total=len(d)) as pbar:
      #   iterator = pool.imap_unordered(method, d.keys(), chunksize=chunksize_inner)
      #   for utterance_id, include in iterator:
      #     transformed_utterances[utterance_id] = include
      #     pbar.update()
      # x = transformed_utterances
    return transformed_utterances


def do_pipeline(utterances: Utterances, selection: Selection, n_jobs: int, chunksize_inner: int, chunksize_outer: int):
  logger = getLogger(__name__)
  acronym_estimator = AcronymEstimator()
  remove_transformer = RemoveTransformer()
  chunking_transformer = ChunkingTransformer()
  dechunking_tranformer = DechunkingTransformer()
  remove_selection_transformer = RemoveSelectionTransformer()

  chunking_transformer.fit()
  acronym_estimator.fit()
  remove_transformer.fit()
  dechunking_tranformer.fit()
  remove_selection_transformer.fit()

  method = partial(
    chunk_pipeline,
    language=utterances.language,
    symbol_format=utterances.symbol_format
  )

  with ProcessPoolExecutor(max_workers=n_jobs) as ex:
    transformed_utterances: List[Any] = dict(tqdm(
      ex.map(method, utterances.items(), chunksize=chunksize_inner),
      total=len(utterances)
    ))

  # with Manager() as manager:
  #   manager = cast(SyncManager, manager)

  #   method = partial(
  #     chunk_pipeline,
  #     language=utterances.language,
  #     symbol_format=utterances.symbol_format
  #   )

  #   logger.info("Copying dict...")
  #   utts = manager.dict(utterances)
  #   logger.info("Copying Done.")

  #   del utterances

  #   with ProcessPoolExecutor(max_workers=n_jobs) as ex:
  #     transformed_utterances: List[Any] = dict(tqdm(
  #       ex.map(method, utts.items(), chunksize=chunksize_inner),
  #       total=len(utts)
  #     ))

  #items = utterances.items()
  #x = list(get_chunks(*items, chunksize=chunksize))
  # for entry in x:
  # e  entries = dict(zip(*entry))
  #  key, items = entry
  #  # print(entry)

  #list_of_utterances = chunking_transformer.transform(utterances, chunksize_outer)
  logger.info("Creating chunks...")
  outer_chunks = _get_chunks(*utterances.items(), chunksize=chunksize_outer)
  logger.info("Done.")

  transformed_utterances = []
  use_mp = True
  if use_mp:

    with Manager() as manager:
      manager = cast(SyncManager, manager)

      cache = manager.dict(utterances)

      method = partial(
        chunk_pipeline,
        language=utterances.language,
        symbol_format=utterances.symbol_format
      )

      for outer_chunk_nr, outer_chunk in enumerate(outer_chunks, start=1):
        logger.info(f"Started chunk {outer_chunk_nr}.")
        outer_chunk_restored = OrderedDict(zip(*outer_chunk))
        logger.info(f"Using {n_jobs} jobs with a chunksize of {chunksize_inner}.")

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
          if False:
            futures: List[Future] = []
            for utt in tqdm(outer_chunk.items()):
              fut = ex.submit(method, utt)
              futures.append(fut)
            for ft in tqdm(futures):
              transformed_utterances.append(ft.result())
          else:
            transformed_utterances: List[Any] = dict(tqdm(
              ex.map(method, outer_chunk_restored.items(), chunksize=chunksize_inner),
              total=len(outer_chunk_restored)
            ))

            # transformed_utterances: List[Utterances] = list(tqdm(
            #   ex.map(chunk_pipeline, list_of_utterances, chunksize=1),
            #   total=len(list_of_utterances)
            # ))
  else:
    for utterances in list_of_utterances:
      acronym_estimator_result = acronym_estimator.estimate(utterances)
      utterances = remove_transformer.transform(
        utterances, acronym_estimator_result)
      transformed_utterances.append(utterances)
  utterances = dechunking_tranformer.transform(transformed_utterances)
  selection = remove_selection_transformer.transform(utterances, selection)
