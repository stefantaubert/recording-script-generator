from collections import Counter, OrderedDict
from logging import getLogger
from typing import Dict, List
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame
from recording_script_generator.core.main import (PreparationData,
                                                  ReadingPassage,
                                                  Representation, SentenceId,
                                                  Symbols)
from text_utils.text import get_ngrams


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


def get_reading_scripts(data: PreparationData) -> Tuple[DataFrame, DataFrame]:
  selected = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in data.selected})
  rest = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k not in data.selected})

  selected_df = get_df_from_reading_passages(selected)
  rest_df = get_df_from_reading_passages(rest)

  return selected_df, rest_df


def df_to_txt(df: DataFrame) -> str:
  result = ""
  for _, row in df.iterrows():
    result += f"{row['Nr']}: {row['Utterance']}\n"
  return result


def df_to_tex(df: DataFrame) -> str:
  result = "\\begin{itemize}\n"
  for _, row in df.iterrows():
    result += f"  \\item[{row['Nr']}:] {row['Utterance']}\n"
  result += "\\end{itemize}"
  return result


def _log_counter(c: Counter):
  logger = getLogger(__name__)
  for char, occ in c.most_common():
    logger.info(f"- {char} ({occ}x)")


def log_ngram_stats(ngrams: List[Tuple[str, ...]]) -> None:
  logger = getLogger(__name__)
  if len(ngrams) == 0:
    return None
  counter = Counter(ngrams)
  total = sum(counter.values())
  for key, value in counter.most_common():
    logger.info(f"- {key}: {value/total*100:.4f}")
  return None


def get_ngram_stats_count(ngrams: List[Tuple[str, ...]], ngrams_order: OrderedSet[Tuple[str, ...]]) -> Tuple[int, ...]:
  counter = Counter(ngrams)
  res = tuple(counter[n_gram] if n_gram in counter else 0 for n_gram in ngrams_order)
  return res


def get_ngram_stats_percent(ngram_stats_count: Tuple[int, ...]) -> Tuple[float, ...]:
  total_count = sum(ngram_stats_count)
  if total_count == 0:
    ngram_stats_percent = tuple(np.nan for _ in ngram_stats_count)
  else:
    ngram_stats_percent = tuple(count / total_count * 100 for count in ngram_stats_count)
  return ngram_stats_percent


def get_n_gram_stats_df(data: Dict[SentenceId, Symbols], selected: OrderedSet[SentenceId], n: int) -> DataFrame:
  columns = [
    f"{n}-gram",
    "Selected #",
    "Not selected #",
    "Total #",
    "Selected %",
    "Not selected %",
    "Total %",
  ]

  if len(data) == 0:
    empty_stats_df = DataFrame(data=[], columns=columns)
    return empty_stats_df

  selected = [utterance_symbols for utterance_id,
              utterance_symbols in data.items() if utterance_id in selected]
  not_selected = [utterance_symbols for utterance_id,
                  utterance_symbols in data.items() if utterance_id not in selected]

  selected_n_grams = [n_gram for utterance_symbols in selected
                      for n_gram in get_ngrams(utterance_symbols, n=n)]
  not_selected_n_grams = [n_gram for utterance_symbols in not_selected
                          for n_gram in get_ngrams(utterance_symbols, n=n)]

  total_n_grams = selected_n_grams + not_selected_n_grams
  total_n_grams_ordered = OrderedSet(list(sorted(total_n_grams)))
  all_n_grams_str = tuple(''.join(n_grams) for n_grams in total_n_grams_ordered)
  total_n_grams_count = get_ngram_stats_count(total_n_grams, total_n_grams_ordered)
  total_n_grams_percent = get_ngram_stats_percent(total_n_grams_count)
  selected_n_grams_count = get_ngram_stats_count(selected_n_grams, total_n_grams_ordered)
  selected_n_grams_percent = get_ngram_stats_percent(selected_n_grams_count)
  not_selected_n_grams_count = get_ngram_stats_count(not_selected_n_grams, total_n_grams_ordered)
  not_selected_n_grams_percent = get_ngram_stats_percent(not_selected_n_grams_count)

  data = np.array([
    all_n_grams_str,
    selected_n_grams_count,
    not_selected_n_grams_count,
    total_n_grams_count,
    selected_n_grams_percent,
    not_selected_n_grams_percent,
    total_n_grams_percent
  ]).T

  n_gram_stats_df = DataFrame(data=data, columns=columns)

  return n_gram_stats_df


def log_general_stats(data: PreparationData, avg_chars_per_s: int) -> None:
  assert avg_chars_per_s >= 0
  logger = getLogger(__name__)

  selected = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in data.selected})
  rest = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k not in data.selected})

  selected_read_chars_len = len([x for (read, rep) in selected.values() for x in read])
  rest_read_chars_len = len([x for (read, rep) in rest.values() for x in read])

  logger.info(
    f"Selected: {len(selected)} entries / {selected_read_chars_len} chars / ca. {selected_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {selected_read_chars_len/avg_chars_per_s/60/60:.2f}h")
  logger.info(
    f"Non-selected: {len(rest)} entries / {rest_read_chars_len} chars / ca. {rest_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {rest_read_chars_len/avg_chars_per_s/60/60:.2f}h")

  selected_chars = {x for (read, rep) in selected.values() for x in rep}
  rest_chars = {x for (read, rep) in rest.values() for x in rep}
  logger.info(f"Selected chars ({len(selected_chars)}):\t{' '.join(sorted(selected_chars))}")
  logger.info(f"Non-selected chars ({len(rest_chars)}):\t{' '.join(sorted(rest_chars))}")
