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
