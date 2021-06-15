from collections import OrderedDict
from dataclasses import dataclass
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from ordered_set import OrderedSet
from recording_script_generator.core.preparation import PreparationData


@dataclass
class ScriptData:
  reading_passages: OrderedDictType[int, List[str]]
  representations: OrderedDictType[int, List[str]]


def merge(data: List[PreparationData]) -> ScriptData:
  entries: List[Tuple[Tuple[str], Tuple[str]]] = []

  for entry in data:
    for reading_passage, representation in zip(entry.reading_passages, entry.representations):
      entries.append(tuple([tuple(reading_passage), tuple(representation)]))

  entries_set = OrderedSet(entries)

  res = ScriptData(
    reading_passages=OrderedDict({i: list(read) for i, (read, _) in enumerate(entries_set)}),
    representations=OrderedDict({i: list(rep) for i, (_, rep) in enumerate(entries_set)}),
  )

  return res
