from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterance, Utterances
from text_utils import text_normalize
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat


def normalize_func(utterance: Utterance, language: Language, symbol_format: SymbolFormat) -> Utterance:
  assert isinstance(utterance, str)
  result = text_normalize(
    text=utterance,
    text_format=symbol_format,
    lang=language,
  )

  return result


def normalize_utterances_inplace(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize_func,
    symbol_format=utterances.symbol_format,
    language=utterances.language,
  )

  result = execute_method_on_utterances_mp(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )
  utterances.update(result)
