from enum import IntEnum
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from ordered_set import OrderedSet
from text_utils import Language, Symbol, SymbolFormat, Symbols


class Mode(IntEnum):
  SELECT = 0
  DESELECT = 1


UtteranceId = int

Utterance = Tuple[UtteranceId, Symbols]


class Utterances(OrderedDictType[UtteranceId, Symbols]):
  symbol_format: SymbolFormat
  language: Language


def clone_utterances(utterances: Utterances) -> Utterances:
  result = utterances.copy()
  result.symbol_format = utterances.symbol_format
  result.language = utterances.language
  return result


class ReadingPassages(Utterances):
  pass


class Representations(Utterances):
  pass


class Selection(OrderedSet):
  pass
