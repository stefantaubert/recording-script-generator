import random
from collections import OrderedDict
from enum import IntEnum
from logging import getLogger
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from ordered_set import OrderedSet
from pandas import DataFrame
from recording_script_generator.core.main import (ReadingPassages,
                                                  Representations, Selection,
                                                  Utterances)
from recording_script_generator.utils import detect_ids_from_tex
from text_selection import greedy_kld_uniform_ngrams_default
from text_selection.greedy_kld_export import greedy_kld_uniform_ngrams_parts
from textgrid.textgrid import Interval, IntervalTier, TextGrid


class SortingMode(IntEnum):
  SYMBOL_COUNT_ASC = 0
  BY_SELECTION = 1
  BY_INDEX = 2
  RANDOM = 3
  KLD = 4
  KLD_PARTS = 5


def number_prepend_zeros(n: int, max_n: int) -> str:
  assert n >= 0
  assert max_n >= 0
  decimals = len(str(max_n))
  res = str(n).zfill(decimals)
  return res


def get_df_from_reading_passages(reading_passages: OrderedDictType[int, Tuple[ReadingPassages, Representations]]) -> DataFrame:
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


def get_keys_custom_sort(metadata: Selection, data: Utterances, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]], parts_count: Optional[int], take_per_part: Optional[int]) -> OrderedSet[int]:

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

  if mode == SortingMode.KLD_PARTS:
    assert ignore_symbols is not None
    assert parts_count is not None
    assert take_per_part is not None

    keys_sorted_by_index = greedy_kld_uniform_ngrams_parts(
      data=selected_representations,
      n_gram=1,
      ignore_symbols=ignore_symbols,
      parts_count=parts_count,
      take_per_part=take_per_part,
    )
    return keys_sorted_by_index

  raise Exception()


def get_reading_scripts(metadata: Selection, data: Utterances, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]], parts_count: Optional[int], take_per_part: Optional[int]) -> Tuple[DataFrame, DataFrame]:
  keys_sorted = get_keys_custom_sort(
    data=data,
    mode=mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    take_per_part=take_per_part,
    parts_count=parts_count,
  )

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


def df_to_consecutive_txt(df: DataFrame) -> str:
  all_utterances = [row['Utterance'] for _, row in df.iterrows()]
  result = " ".join(all_utterances)
  return result


def df_to_tex(df: DataFrame, use_hint_on_question_and_exclamation: bool = True) -> str:
  result = "\\begin{enumerate}[label={\\protect\\threedigits{\\theenumi}:}]\n"
  for _, row in df.iterrows():
    hint = ""
    if use_hint_on_question_and_exclamation:
      add_hint = len(row["Utterance"]) > 0 and row["Utterance"][-1] in {"!", "?"}
      if add_hint:
        hint = f"{row['Utterance'][-1]} "
    result += f"  \\item {hint}{row['Utterance']} % {row['Id']}\n"
  result += "\\end{enumerate}"
  return result


def generate_textgrid(metadata: Selection, data: Utterances, tex: str, reading_speed_chars_per_s: float) -> TextGrid:
  ids_in_tex = detect_ids_from_tex(tex)
  grid = TextGrid(
    name="reading passages",
    minTime=0,
    maxTime=None,
    strict=True,
  )

  graphemes_tier = IntervalTier(
    name="sentences (graphemes)",
    minTime=0,
    maxTime=None,
  )
  grid.append(graphemes_tier)

  phonemes_tier = IntervalTier(
    name="sentences (phonemes)",
    minTime=0,
    maxTime=None,
  )
  grid.append(phonemes_tier)

  last_time = 0
  for read_id in ids_in_tex:
    duration = len(data.reading_passages[read_id]) / reading_speed_chars_per_s
    min_time = last_time
    max_time = last_time + duration

    graphemes = ''.join(data.reading_passages[read_id])
    graphemes_interval = Interval(
      minTime=min_time,
      maxTime=max_time,
      mark=graphemes,
    )
    graphemes_tier.addInterval(graphemes_interval)

    phonemes = ''.join(data.representations[read_id])
    phonemes_interval = Interval(
      minTime=min_time,
      maxTime=max_time,
      mark=phonemes,
    )
    phonemes_tier.addInterval(phonemes_interval)

    last_time = max_time

  grid.maxTime = last_time
  graphemes_tier.maxTime = last_time
  phonemes_tier.maxTime = last_time

  return grid
