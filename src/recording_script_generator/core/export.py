import random
import re
from collections import OrderedDict
from enum import IntEnum
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from ordered_set import OrderedSet
from pandas import DataFrame
from recording_script_generator.core.main import (PreparationData,
                                                  ReadingPassage,
                                                  ReadingPassages,
                                                  Representation)
from text_selection import greedy_kld_uniform_ngrams_default

IDS_TEX_PATTERN = re.compile(" % ([0-9]*)\n")


class SortingMode(IntEnum):
  SYMBOL_COUNT_ASC = 0
  BY_SELECTION = 1
  BY_INDEX = 2
  RANDOM = 3
  KLD = 4


def number_prepend_zeros(n: int, max_n: int) -> str:
  assert n >= 0
  assert max_n >= 0
  decimals = len(str(max_n))
  res = str(n).zfill(decimals)
  return res


def get_df_from_reading_passages(reading_passages: OrderedDictType[int, Tuple[ReadingPassage, Representation]]) -> DataFrame:
  df = DataFrame(
    data=[(
      k,
      number_prepend_zeros(i + 1, len(reading_passages) + 1),
      "".join(reading_passage),
      "".join(representation),
    ) for i, (k, (reading_passage, representation)) in enumerate(reading_passages.items())],
    columns=["Id", "Nr", "Utterance", "Representation"],
  )

  return df


def get_keys_custom_sort(data: PreparationData, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]]) -> OrderedSet[int]:

  selected_reading_passages = OrderedDict({k: data.reading_passages[k] for k in data.selected})

  if mode == SortingMode.RANDOM:
    assert seed is not None
    keys = list(selected_reading_passages.keys())
    random.seed(seed)
    random.shuffle(keys)
    result = OrderedSet(keys)
    return result

  if mode == SortingMode.BY_SELECTION:
    unchanged_keys = OrderedSet(list(selected_reading_passages.keys()))
    return unchanged_keys

  if mode == SortingMode.SYMBOL_COUNT_ASC:
    reading_passages_lens = [(key, len(symbols))
                             for key, symbols in selected_reading_passages.items()]
    reading_passages_lens.sort(key=lambda key_lens: key_lens[1])
    result = OrderedSet([key for key, _ in reading_passages_lens])
    return result

  if mode == SortingMode.BY_INDEX:
    keys_sorted_by_index = OrderedSet(list(sorted(selected_reading_passages.keys())))
    return keys_sorted_by_index

  selected_representations = OrderedDict({k: data.representations[k] for k in data.selected})

  if mode == SortingMode.KLD:
    assert ignore_symbols is not None
    keys_sorted_by_index = greedy_kld_uniform_ngrams_default(
      data=selected_representations,
      n_gram=1,
      ignore_symbols=ignore_symbols
    )
    return keys_sorted_by_index

  raise Exception()


def get_reading_scripts(data: PreparationData, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]]) -> Tuple[DataFrame, DataFrame]:
  keys_sorted = get_keys_custom_sort(data, mode=mode, seed=seed, ignore_symbols=ignore_symbols)
  selected = OrderedDict(
    {k: (data.reading_passages[k], data.representations[k])
     for k in keys_sorted})
  rest = OrderedDict(
    {k: (data.reading_passages[k], data.representations[k])
     for k in data.reading_passages if k not in keys_sorted})

  selected_df = get_df_from_reading_passages(selected)
  rest_df = get_df_from_reading_passages(rest)

  return selected_df, rest_df


def df_to_txt(df: DataFrame) -> str:
  result = ""
  for _, row in df.iterrows():
    result += f"{row['Nr']}: {row['Utterance']}\n"
  return result


def df_to_tex(df: DataFrame) -> str:
  result = "\\begin{enumerate}[label={\\protect\\threedigits{\\theenumi}:}]\n"
  for _, row in df.iterrows():
    result += f"  \\item {row['Utterance']} % {row['Id']}\n"
  result += "\\end{enumerate}"
  return result


def detect_ids_from_tex(tex: str) -> OrderedSet[int]:
  res = OrderedSet()
  matches = re.findall(IDS_TEX_PATTERN, tex)
  for match in matches:
    res.add(int(match))
  return res
