import itertools
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import Manager, managers, synchronize
from multiprocessing.managers import SyncManager
from sys import getsizeof
from time import sleep
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
  logger.info(f"Size of utterances: {getsizeof(utterances)}")
  with Manager() as manager:
    manager = cast(SyncManager, manager)
    # ignore order
    #x = manager.list(list(utterances.items()))
    d = manager.dict({k: v for k, v in utterances.items()})
    #manager.Array()
    logger.info(f"Size of utterances in manager: {getsizeof(d)}")
    del utterances

    method = partial(
      chunk_pipeline_v2,
      utterances=d,
      language=language,
      symbol_format=symbol_format
    )

    utterance_ids = list(d.keys())

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
      transformed_utterances: List[Any] = dict(tqdm(
        ex.map(method, utterance_ids, chunksize=chunksize_inner),
        total=len(utterance_ids)
      ))
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
