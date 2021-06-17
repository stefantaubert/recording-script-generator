from collections import Counter, OrderedDict
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from numpy import select
from ordered_set import OrderedSet
from pandas import DataFrame
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


def number_prepend_zeros(n: int, max_n: int) -> str:
  assert n >= 0
  assert max_n >= 0
  decimals = len(str(max_n))
  res = str(n).zfill(decimals)
  return res


def get_df_from_reading_passages(reading_passages: OrderedDictType[int, Tuple[List[str], List[str]]]) -> DataFrame:
  df = DataFrame(
    data=[(
      k,
      number_prepend_zeros(i + 1, len(reading_passages) + 1),
      "".join(reading_passage),
      "".join(representation),
    ) for i, (k, (reading_passage, representation)) in enumerate(reading_passages.items())],
    columns=["id", "nr", "utterance", "representation"],
  )

  return df


def get_reading_scripts(data: ScriptData, selection: Selection) -> Tuple[DataFrame, DataFrame, DataFrame]:
  selected = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.selected})
  ignored = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.ignored})
  rest = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.rest})

  selected_df = get_df_from_reading_passages(selected)
  ignored_df = get_df_from_reading_passages(ignored)
  rest_df = get_df_from_reading_passages(rest)

  return selected_df, ignored_df, rest_df


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
  rest = OrderedDict({k: v for k, v in data.representations.items() if k in selection.rest})
  new_selected = greedy_ngrams_epochs(
    data=rest,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )

  result = Selection(
    selected=selection.selected | new_selected,
    ignored=selection.ignored,
    rest=selection.rest - new_selected,
  )

  logger.info(f"Added {len(new_selected)} utterances to selection.")

  return result


def select_rest(selection: Selection) -> Selection:
  logger = getLogger(__name__)
  result = Selection(
    selected=selection.selected | selection.rest,
    ignored=selection.ignored,
    rest=OrderedDict(),
  )

  logger.info(f"Added {len(selection.rest)} utterances to selection.")

  return result


def merge_merged(merged_data: List[Tuple[ScriptData, Selection]]) -> Tuple[ScriptData, Selection]:
  reading_passages: OrderedDictType[int, List[str]] = OrderedDict()
  representations: OrderedDictType[int, List[str]] = OrderedDict()
  selected: OrderedSet[int] = OrderedSet()
  ignored: OrderedSet[int] = OrderedSet()
  rest: OrderedSet[int] = OrderedSet()

  offset = 0
  for script, selection in merged_data:
    copied_reading_passages = OrderedDict(
      {k + offset: v for k, v in script.reading_passages.items()})
    copied_representations = OrderedDict(
      {k + offset: v for k, v in script.representations.items()})
    reading_passages.update(copied_reading_passages)
    representations.update(copied_representations)
    selected |= OrderedSet([k + offset for k in selection.selected])
    ignored |= OrderedSet([k + offset for k in selection.ignored])
    rest |= OrderedSet([k + offset for k in selection.rest])
    offset += len(copied_reading_passages)

  res_data = ScriptData(
    reading_passages=reading_passages,
    representations=representations,
  )

  res_selection = Selection(
    selected=selected,
    ignored=ignored,
    rest=rest,
  )

  return res_data, res_selection


def ignore(data: ScriptData, selection: Selection, ignore_symbol: Optional[str]) -> Selection:
  logger = getLogger(__name__)
  rest = OrderedDict({k: v for k, v in data.representations.items() if k in selection.rest})

  ignore = OrderedSet()
  for k, v in rest.items():
    if ignore_symbol is not None and ignore_symbol in v:
      ignore.add(k)

  result = Selection(
    selected=selection.selected,
    ignored=selection.ignored | ignore,
    rest=selection.rest - ignore,
  )

  logger.info(f"Ignored {len(ignore)} utterances.")

  return result


def _log_counter(c: Counter):
  logger = getLogger(__name__)
  for char, occ in c.most_common():
    logger.info(f"- {char} ({occ}x)")


def log_stats(data: ScriptData, selection: Selection, avg_chars_per_s: int) -> None:
  assert avg_chars_per_s >= 0
  counter_repr = Counter([x for y in data.representations.values() for x in y])
  counter_read = Counter([x for y in data.reading_passages.values() for x in y])

  logger = getLogger(__name__)
  logger.info("Representation symbol occurrences:")
  _log_counter(counter_repr)
  logger.info("Reading passages symbol occurrences:")
  _log_counter(counter_read)

  selected = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.selected})
  ignored = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.ignored})
  rest = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in selection.rest})

  selected_read_chars_len = len([x for (read, rep) in selected.values() for x in read])
  ignored_read_chars_len = len([x for (read, rep) in ignored.values() for x in read])
  rest_read_chars_len = len([x for (read, rep) in rest.values() for x in read])

  logger.info(
    f"Selected: {len(selected)} entries / {selected_read_chars_len} chars / ~{selected_read_chars_len/avg_chars_per_s/60:.2f} min / ~{selected_read_chars_len/avg_chars_per_s/60/60:.2f} h")
  logger.info(
    f"Ignored: {len(ignored)} entries / {ignored_read_chars_len} chars / ~{ignored_read_chars_len/avg_chars_per_s/60:.2f} min / ~{ignored_read_chars_len/avg_chars_per_s/60/60:.2f} h")
  logger.info(
    f"Rest: {len(rest)} entries / {rest_read_chars_len} chars / ~{rest_read_chars_len/avg_chars_per_s/60:.2f} min / ~{rest_read_chars_len/avg_chars_per_s/60/60:.2f} h")

  selected_chars = {x for (read, rep) in selected.values() for x in rep}
  ignored_chars = {x for (read, rep) in ignored.values() for x in rep}
  rest_chars = {x for (read, rep) in rest.values() for x in rep}
  logger.info(f"Selected chars ({len(selected_chars)}):\t{' '.join(list(selected_chars))}")
  logger.info(f"Ignored chars ({len(ignored_chars)}):\t{' '.join(list(ignored_chars))}")
  logger.info(f"Rest chars ({len(rest_chars)}):\t{' '.join(list(rest_chars))}")
