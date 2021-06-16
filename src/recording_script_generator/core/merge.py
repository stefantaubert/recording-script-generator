from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple
from logging import getLogger

from ordered_set import OrderedSet
from recording_script_generator.core.preparation import PreparationData
from text_selection import greedy_ngrams_epochs


@dataclass
class ScriptData:
  reading_passages: OrderedDictType[int, List[str]]
  representations: OrderedDictType[int, List[str]]

@dataclass
class Selection:
  selected: OrderedSet[int]
  ignored: OrderedSet[int]
  rest: OrderedSet[int]

def merge(data: List[PreparationData]) -> Tuple[ScriptData, Selection]:
  entries: List[Tuple[Tuple[str], Tuple[str]]] = []

  for entry in data:
    for reading_passage, representation in zip(entry.reading_passages, entry.representations):
      entries.append(tuple([tuple(reading_passage), tuple(representation)]))

  entries_set = OrderedSet(entries)

  res = ScriptData(
    reading_passages=OrderedDict({i: list(read) for i, (read, _) in enumerate(entries_set)}),
    representations=OrderedDict({i: list(rep) for i, (_, rep) in enumerate(entries_set)}),
  )

  assert res.reading_passages.keys() == res.representations.keys()

  selection = Selection(
    selected=OrderedSet(),
    ignored=OrderedSet(),
    rest=OrderedSet(res.reading_passages.keys()),
  )

  return res, selection


def select_greedy_ngrams_epochs(data: ScriptData, selection: Selection, n_gram: int, epochs: int, ignore_symbols: Optional[Set[str]]) -> Selection:
  logger = getLogger(__name__)
  rest = OrderedDict({k:v for k,v in data.representations.items() if k in selection.rest})
  new_selected = greedy_ngrams_epochs(
    data=rest,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs = epochs,
  )

  result = Selection(
    selected=selection.selected | new_selected,
    ignored=selection.ignored,
    rest=selection.rest - new_selected,
  )

  logger.info(f"Added {len(new_selected)} utterances to selection.")

  return result
