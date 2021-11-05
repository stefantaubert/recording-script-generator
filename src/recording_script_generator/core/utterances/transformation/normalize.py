from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterances
from text_utils import text_normalize, text_to_symbols
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Symbols


def normalize_func(symbols: Symbols, language: Language, symbol_format: SymbolFormat) -> Symbols:
  symbols_str = ''.join(symbols)
  result = text_normalize(
    text=symbols_str,
    text_format=symbol_format,
    lang=language,
  )

  sentences = text_to_symbols(
    text=result,
    lang=language,
    text_format=symbol_format,
  )

  return sentences


def normalize_utterances_inplace(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> None:
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
  )
  utterances.update(result)
