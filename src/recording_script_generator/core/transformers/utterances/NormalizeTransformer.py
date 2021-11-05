from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import (Utterance, Utterances,
                                                   clone_utterances)
from text_utils import text_normalize, text_to_symbols
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Symbols
from tqdm import tqdm


def get_utterances_normalized(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Utterances:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    normalize_func,
    text_format=utterances.text_format,
    language=utterances.language,
  )

  result = Utterances(execute_method_on_utterances_mp(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  ))

  result.language = utterances.language
  result.symbol_format = utterances.symbol_format

  return result


def normalize_func(symbols: Symbols, lang: Language, text_format: SymbolFormat) -> Symbols:
  symbols_str = ''.join(symbols)
  result = text_normalize(
    text=symbols_str,
    text_format=text_format,
    lang=lang,
  )

  sentences = text_to_symbols(
    text=result,
    lang=lang,
    text_format=text_format,
  )

  return sentences


class NormalizeTransformer():
  def fit(self):
    pass

  def transform(self, utterances: Utterances) -> Utterances:
    logger = getLogger(__name__)
    logger.info("Normalizing...")
    result = clone_utterances(utterances)
    for utterance in tqdm(utterances.items()):
      utterance_id, symbols = normalize_func(
        utterance, utterances.language, utterances.symbol_format)
      result[utterance_id] = symbols
    return result
