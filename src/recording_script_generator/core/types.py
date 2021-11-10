from logging import getLogger
from sys import getsizeof
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Union

from ordered_set import OrderedSet
from text_utils import Language, SymbolFormat, Symbols, text_to_symbols
from tqdm import tqdm

Selection = OrderedSet

UtteranceId = int

UtteranceKVPair = Tuple[UtteranceId, Symbols]
Utterance = Union[Symbols, str]


class Utterances(OrderedDictType[UtteranceId, Utterance]):
  symbol_format: SymbolFormat
  language: Language


ReadingPassages = Utterances
Representations = Utterances


def utterance_to_str(utterance: Utterance) -> str:
  if isinstance(utterance, str):
    return utterance
  elif isinstance(utterance, tuple):
    return ''.join(utterance)
  assert False


def utterance_to_symbols(utterance: Utterance, text_format: Optional[SymbolFormat], language: Optional[Language]) -> Symbols:
  if isinstance(utterance, tuple):
    return utterance
  elif isinstance(utterance, str):
    assert language is not None
    assert text_format is not None
    return text_to_symbols(
      text=utterance,
      text_format=text_format,
      lang=language,
    )
  assert False


def get_utterance_duration_s(utterance: Utterance, reading_speed_chars_per_s: float) -> float:
  duration = len(utterance) / reading_speed_chars_per_s
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
