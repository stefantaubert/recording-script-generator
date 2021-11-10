from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import (Utterance, Utterances,
                                                   utterance_to_symbols)
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat


def main(utterance: Utterance, language: Language, symbol_format: SymbolFormat) -> str:
  symbols = utterance_to_symbols(
    utterance=utterance,
    language=language,
    text_format=symbol_format,
  )

  return symbols


def convert_to_symbols_inplace(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to symbols...")
  method = partial(
    main,
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
