from collections import Counter, OrderedDict
from logging import getLogger
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from pandas import DataFrame
from recording_script_generator.core.main import (PreparationData,
                                                  ReadingPassage,
                                                  Representation)


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


def _log_counter(c: Counter):
  logger = getLogger(__name__)
  for char, occ in c.most_common():
    logger.info(f"- {char} ({occ}x)")


def log_stats(data: PreparationData, avg_chars_per_s: int) -> None:
  assert avg_chars_per_s >= 0
  counter_repr = Counter([x for y in data.representations.values() for x in y])
  counter_read = Counter([x for y in data.reading_passages.values() for x in y])

  logger = getLogger(__name__)
  logger.info("Representation symbol occurrences:")
  _log_counter(counter_repr)
  logger.info("Reading passages symbol occurrences:")
  _log_counter(counter_read)

  selected = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k in data.selected})
  rest = OrderedDict(
    {k: (v, data.representations[k]) for k, v in data.reading_passages.items() if k not in data.selected})

  selected_read_chars_len = len([x for (read, rep) in selected.values() for x in read])
  rest_read_chars_len = len([x for (read, rep) in rest.values() for x in read])

  logger.info(
    f"Selected: {len(selected)} entries / {selected_read_chars_len} chars / ca. {selected_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {selected_read_chars_len/avg_chars_per_s/60/60:.2f}h")
  logger.info(
    f"Rest: {len(rest)} entries / {rest_read_chars_len} chars / ca. {rest_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {rest_read_chars_len/avg_chars_per_s/60/60:.2f}h")

  selected_chars = {x for (read, rep) in selected.values() for x in rep}
  rest_chars = {x for (read, rep) in rest.values() for x in rep}
  logger.info(f"Selected chars ({len(selected_chars)}):\t{' '.join(sorted(selected_chars))}")
  logger.info(f"Rest chars ({len(rest_chars)}):\t{' '.join(sorted(rest_chars))}")
