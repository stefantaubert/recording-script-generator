from logging import getLogger

from recording_script_generator.core.types import (Utterance, Utterances,
                                                   clone_utterances)
from text_utils import text_normalize, text_to_symbols
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from tqdm import tqdm


def normalize_func(utterance: Utterance, lang: Language, text_format: SymbolFormat) -> Utterance:
  utterance_id, symbols = utterance
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

  return utterance_id, sentences


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