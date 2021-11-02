import itertools
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from logging import getLogger
from time import sleep
from typing import Any, Dict, List, Tuple

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
from text_utils.types import Symbols
from tqdm import tqdm


def chunk_pipeline(utterance: Utterance, language: Language, symbol_format: SymbolFormat):

  logger = getLogger(__name__)
  # logger.info("start")

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


def do_pipeline(utterances: Utterances, selection: Selection, n_jobs: int, chunksize: int):
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

  #items = utterances.items()
  #x = list(get_chunks(*items, chunksize=chunksize))
  # for entry in x:
  # e  entries = dict(zip(*entry))
  #  key, items = entry
  #  # print(entry)

  # list_of_utterances = chunking_transformer.transform(utterances, chunksize)

  transformed_utterances = []
  use_mp = True
  if use_mp:
    logger.info(f"Using {n_jobs} jobs with a chunksize of {chunksize}.")
    method = partial(
      chunk_pipeline,
      language=utterances.language,
      symbol_format=utterances.symbol_format
    )

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
      if False:
        futures: List[Future] = []
        for utt in list_of_utterances:
          fut = ex.submit(chunk_pipeline, utt)
          futures.append(fut)
        for ft in futures:
          transformed_utterances.append(ft.result())
      else:
        transformed_utterances: List[Any] = dict(tqdm(
          ex.map(method, utterances.items(), chunksize=chunksize),
          total=len(utterances)
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
