import random
from collections import OrderedDict
from enum import IntEnum
from logging import getLogger
from multiprocessing import cpu_count
from typing import Optional, Set

from ordered_set import OrderedSet
from pandas import DataFrame
from recording_script_generator.core.selection.detection.tex import \
    detect_ids_from_tex
from recording_script_generator.core.types import (Paths, ReadingPassages,
                                                   ReadingPassagesPaths,
                                                   Representations,
                                                   UtteranceId,
                                                   get_utterance_duration_s,
                                                   utterance_to_symbols,
                                                   utterance_to_text)
from text_selection import greedy_kld_uniform_ngrams_default
from text_selection.greedy_kld_export import greedy_kld_uniform_ngrams_parts
from textgrid.textgrid import Interval, IntervalTier, TextGrid
from tqdm import tqdm

COL_ID = "Id"
COL_NR = "Nr"
COL_UTTERANCE = "Utterance"
COL_REPRESENTATION = "Representation"
COL_PATH = "Path"


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


def get_df(selection: OrderedSet[UtteranceId], reading_passages: ReadingPassages, representations: Representations, paths: Paths, read_paths: ReadingPassagesPaths, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]], parts_count: Optional[int], take_per_part: Optional[int]) -> DataFrame:
  selected_keys_sorted = __get_keys_custom_sort(
    selection=selection,
    reading_passages=reading_passages,
    representations=representations,
    mode=mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    take_per_part=take_per_part,
    parts_count=parts_count,
  )

  logger = getLogger(__name__)
  df_data = (
    (
      k,
      number_prepend_zeros(i, len(selected_keys_sorted)),
      utterance_to_text(reading_passages[k]),
      utterance_to_text(representations[k]),
      str(paths[read_paths[k]]),
    )
    for i, k in enumerate(selected_keys_sorted, start=1)
  )

  logger.info("Creating content...")
  df = DataFrame(
    data=tqdm(df_data),
    columns=(COL_ID, COL_NR, COL_UTTERANCE, COL_REPRESENTATION, COL_PATH),
  )

  return df


def __get_keys_custom_sort(selection: OrderedSet[UtteranceId], reading_passages: ReadingPassages, representations: Representations, mode: SortingMode, seed: Optional[int], ignore_symbols: Optional[Set[str]], parts_count: Optional[int], take_per_part: Optional[int]) -> OrderedSet[UtteranceId]:
  selected_reading_passages = OrderedDict([(k, reading_passages[k]) for k in selection])

  if mode == SortingMode.RANDOM:
    assert seed is not None
    keys = list(selected_reading_passages.keys())
    random.seed(seed)
    random.shuffle(keys)
    result = OrderedSet(keys)
    return result

  if mode == SortingMode.BY_SELECTION:
    unchanged_keys = OrderedSet(selected_reading_passages.keys())
    return unchanged_keys

  if mode == SortingMode.SYMBOL_COUNT_ASC:
    logger = getLogger(__name__)
    logger.info("Sorting keys after symbol count...")
    reading_passages_lens = [(key, len(symbols))
                             for key, symbols in tqdm(selected_reading_passages.items())]
    reading_passages_lens.sort(key=lambda key_lens: key_lens[1])
    result = OrderedSet((key for key, _ in reading_passages_lens))
    logger.info("Done.")
    return result

  if mode == SortingMode.BY_INDEX:
    keys_sorted_by_index = OrderedSet(sorted(selected_reading_passages.keys()))
    return keys_sorted_by_index

  selected_representations = OrderedDict((
    (k, utterance_to_symbols(representations[k]))
    for k in selection
  ))

  if mode == SortingMode.KLD:
    assert ignore_symbols is not None
    keys_sorted_by_index = greedy_kld_uniform_ngrams_default(
      data=selected_representations,
      n_gram=1,
      ignore_symbols=ignore_symbols,
      chunksize=None,
      maxtasksperchild=None,
      n_jobs=cpu_count(),
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
      chunksize=None,
      maxtasksperchild=None,
      n_jobs=cpu_count(),
    )
    return keys_sorted_by_index

  raise Exception()


def df_to_txt(df: DataFrame) -> str:
  result = ""
  for _, row in tqdm(df.iterrows(), total=len(df.index)):
    result += f"{row[COL_NR]}: {row[COL_UTTERANCE]}\n"
  return result.strip("\n")


def df_to_consecutive_txt(df: DataFrame) -> str:
  all_utterances = [row[COL_UTTERANCE] for _, row in tqdm(df.iterrows(), total=len(df.index))]
  result = " ".join(all_utterances)
  return result


def df_to_tex(df: DataFrame, use_hint_on_question_and_exclamation: bool = True) -> str:
  result = "\\begin{enumerate}[label={\\protect\\threedigits{\\theenumi}:}]\n"
  for _, row in tqdm(df.iterrows(), total=len(df.index)):
    hint = ""
    if use_hint_on_question_and_exclamation:
      add_hint = len(row[COL_UTTERANCE]) > 0 and row[COL_UTTERANCE][-1] in {"!", "?"}
      if add_hint:
        hint = f"{row[COL_UTTERANCE][-1]} "
    result += f"  \\item {hint}{row[COL_UTTERANCE]} % {row[COL_ID]}\n"
  result += "\\end{enumerate}"
  return result


def df_to_paths(df: DataFrame) -> str:
  result = ""
  for _, row in tqdm(df.iterrows(), total=len(df.index)):
    result += f"{row[COL_PATH]}\n"
  return result.strip("\n")


def generate_textgrid(reading_passages: ReadingPassages, representations: Representations, tex: str, reading_speed_chars_per_s: float) -> TextGrid:
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
    duration = get_utterance_duration_s(reading_passages[read_id], reading_speed_chars_per_s)
    min_time = last_time
    max_time = last_time + duration

    graphemes = utterance_to_text(reading_passages[read_id])
    graphemes_interval = Interval(
      minTime=min_time,
      maxTime=max_time,
      mark=graphemes,
    )
    graphemes_tier.addInterval(graphemes_interval)

    phonemes = utterance_to_text(representations[read_id])
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
