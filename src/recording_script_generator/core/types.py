from enum import IntEnum
from typing import OrderedDict as OrderedDictType

from ordered_set import OrderedSet
from text_utils import Language, Symbol, SymbolFormat, Symbols


class Mode(IntEnum):
  SELECT = 0
  DESELECT = 1


UtteranceId = int


class Utterances(OrderedDictType[UtteranceId, Symbols]):
  symbol_format: SymbolFormat
  language: Language


class ReadingPassages(Utterances):
  pass


class Representations(Utterances):
  pass


class Selection(OrderedSet):
  pass
