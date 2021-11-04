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

from recording_script_generator.core.estimators.selection import *
from recording_script_generator.core.estimators.utterances import *
from recording_script_generator.core.transformers.utterances import (
    ArpaToIpaTransformer, ChangeIpaTransformer, ChangeTextTransformer,
    EngToArpaTransformer, NormalizeTransformer, RemoveTransformer)
from recording_script_generator.core.types import (Selection, Utterance,
                                                   UtteranceId, Utterances)
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import SymbolId, Symbols
from tqdm import tqdm


def handle_error(exception: Exception) -> None:
  logger = getLogger(__name__)
  #tb = sys.exc_info()
  # traceback.print_stack()
  # print(traceback.format_exc())
  logger.exception(exception)
  remote_traceback = cast(RemoteTraceback, exception.__cause__)
  logger.info(remote_traceback.tb)
  pass


def do_pipeline(utterances: Utterances, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: int):
  logger = getLogger(__name__)

  logger.info(f"Size of utterances in memory: {getsizeof(utterances)/1024**3:.2f} Gb")

  step = AcronymEstimator()
  step.fit(n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = NormalizeTransformer()
  step.fit()
  utterances = step.transform(utterances)

  step = DuplicateEstimator()
  step.fit()
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = UnfrequentWordCountEstimator()
  step.fit(2, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = ProperNameEstimator()
  step.fit(n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = UndesiredTextEstimator()
  undesired = set("/ \\ - : @ ; * % \" ( ) [ ] { } quote oswald ye hath pp.".split(" "))
  step.fit(undesired, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = WordCountEstimator()
  step.fit(3, None, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = UnknownWordEstimator()
  step.fit(0, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = EngToArpaTransformer()
  step.fit(utterances, n_jobs, chunksize)
  utterances = step.transform(utterances)

  step = UndesiredTextEstimator()
  undesired = {"'"}
  step.fit(undesired, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(utterances)

  step = RemoveTransformer()
  step.fit()
  utterances = step.transform(utterances, remove)

  step = ArpaToIpaTransformer()
  step.fit(n_jobs, chunksize)
  utterances = step.transform(utterances)

  return utterances, selection
