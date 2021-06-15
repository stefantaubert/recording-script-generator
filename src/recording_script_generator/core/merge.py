from collections import OrderedDict
from dataclasses import dataclass
from typing import List
from typing import OrderedDict as OrderedDictType

from ordered_set import OrderedSet
from recording_script_generator.core.preparation import PreparationData


@dataclass
class ScriptData:
  reading_passages: OrderedDictType[int, List[str]]
  representations: OrderedDictType[int, List[str]]


def merge(data: OrderedSet[PreparationData]) -> ScriptData:
  reading_passages: OrderedSet[str] = OrderedSet()
  representations: OrderedSet[str] = OrderedSet()

  for entry in data:
    reading_passages |= OrderedSet(entry.reading_passages.values())
    representations |= OrderedSet(entry.representations.values())

  reading_passages |= OrderedSet(entry.reading_passages.values())
  representations |= OrderedSet(entry.representations.values())

  res = ScriptData(
    reading_passages=OrderedDict({i: list(v) for i, v in enumerate(reading_passages)}),
    representations=OrderedDict({i: list(v) for i, v in enumerate(representations)}),
  )

  return res
