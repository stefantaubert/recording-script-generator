from typing import OrderedDict as OrderedDictType
from typing import Tuple

from ordered_set import OrderedSet
from text_utils import Language, SymbolFormat, Symbols

Selection = OrderedSet

UtteranceId = int

Utterance = Tuple[UtteranceId, Symbols]


class Utterances(OrderedDictType[UtteranceId, Symbols]):
  symbol_format: SymbolFormat
  language: Language


ReadingPassages = Utterances
Representations = Utterances
