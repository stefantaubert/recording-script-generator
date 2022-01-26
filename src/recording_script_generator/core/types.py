from pathlib import Path
from typing import Optional
from typing import OrderedDict as OrderedDictType

from ordered_set import OrderedSet
from text_utils import (Language, StringFormat, SymbolFormat, Symbols,
                        SymbolsString)

Selection = OrderedSet

UtteranceId = int
PathId = int

Utterance = SymbolsString
Paths = OrderedDictType[PathId, Optional[Path]]

UtterancesPaths = OrderedDictType[UtteranceId, PathId]
ReadingPassagesPaths = UtterancesPaths


class Utterances(OrderedDictType[UtteranceId, Utterance]):
  entries: OrderedDictType[UtteranceId, Utterance]
  symbol_format: SymbolFormat
  language: Language


ReadingPassages = Utterances
Representations = Utterances


def utterance_to_text(utterance: Utterance) -> str:
  symbols = StringFormat.SYMBOLS.convert_string_to_symbols(utterance)
  text = StringFormat.TEXT.convert_symbols_to_string(symbols)
  del symbols
  return text


def utterance_to_symbols(utterance: Utterance) -> Symbols:
  symbols = StringFormat.SYMBOLS.convert_string_to_symbols(utterance)
  return symbols


def get_utterance_duration_s(utterance: Utterance, reading_speed_chars_per_s: float) -> float:
  symbols = utterance_to_symbols(utterance)
  duration = len(symbols) / reading_speed_chars_per_s
  del symbols
  return duration

# def utterances_to_str_inplace(utterances: Utterances) -> None:
#   for utterance_id, symbols in tqdm(utterances.items()):
#     assert isinstance(symbols, tuple)
#     utterances[utterance_id] = ''.join(symbols)

#   if __debug__:
#     logger = getLogger(__name__)
#     logger.info("Calculating size in memory...")
#     size = getsizeof(utterances)
#     #from pympler import asizeof
#     #size = asizeof.asizeof(res)
#     logger.info(f"Size in memory: {size/1024**3:.2f} Gb.")


# def str_to_utterances_inplace(utterances: Utterances) -> None:
#   for utterance_id, symbols_str in tqdm(utterances.items()):
#     assert isinstance(symbols_str, str)
#     utterances[utterance_id] = tuple(symbols_str)

#   if __debug__:
#     logger = getLogger(__name__)
#     logger.info("Calculating size in memory...")
#     size = getsizeof(utterances)
#     #from pympler import asizeof
#     #size = asizeof.asizeof(res)
#     logger.info(f"Size in memory: {size/1024**3:.2f} Gb.")
