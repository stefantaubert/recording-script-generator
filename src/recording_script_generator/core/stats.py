from collections import Counter, OrderedDict
from logging import getLogger
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   Utterance, Utterances,
                                                   get_utterance_duration_s,
                                                   utterance_to_text,
                                                   utterance_to_symbols)
from text_utils import get_ngrams
from tqdm import tqdm


def log_general_stats(selection: Selection, reading_passages: ReadingPassages, representations: Representations, avg_chars_per_s: float) -> None:
  assert avg_chars_per_s > 0
  logger = getLogger(__name__)

  selected_reading_passages = [v for k, v in tqdm(reading_passages.items()) if k in selection]
  selected_representations = [v for k, v in tqdm(representations.items()) if k in selection]
  deselected_reading_passages = [v for k, v in tqdm(reading_passages.items()) if k not in selection]
  deselected_representations = [v for k, v in tqdm(representations.items()) if k not in selection]

  selected_read_chars_len = sum([1 for read in selected_reading_passages for _ in read])
  deselected_read_chars_len = sum([1 for read in deselected_reading_passages for _ in read])

  selected_word_counts = __get_utterances_word_counts(selected_reading_passages)
  deselected_word_counts = __get_utterances_word_counts(
    deselected_reading_passages)
  selected_durations_s = __get_utterances_durations(selected_reading_passages, avg_chars_per_s)
  deselected_durations_s = __get_utterances_durations(deselected_reading_passages, avg_chars_per_s)

  selected_durations_s_distribution = __get_utterance_duration_distribution(selected_durations_s)
  non_selected_durations_s_distribution = __get_utterance_duration_distribution(
    deselected_durations_s)

  if len(selected_durations_s_distribution) > 0:
    logger.info("Duration distribution for selected utterances:")
    __log_distribution(selected_durations_s_distribution)

  if len(non_selected_durations_s_distribution) > 0:
    logger.info("Duration distribution for deselected utterances:")
    __log_distribution(non_selected_durations_s_distribution)

  logger.info(
    f"Selected: {len(selected_reading_passages)} entries / {selected_read_chars_len} chars / ca. {selected_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {selected_read_chars_len/avg_chars_per_s/60/60:.2f}h")
  logger.info(
    f"Deselected: {len(deselected_reading_passages)} entries / {deselected_read_chars_len} chars / ca. {deselected_read_chars_len/avg_chars_per_s/60:.2f}min / ca. {deselected_read_chars_len/avg_chars_per_s/60/60:.2f}h")

  selected_chars = {x for rep in tqdm(selected_representations) for x in rep}
  rest_chars = {x for rep in tqdm(deselected_representations) for x in rep}
  if len(selected_chars) > 0:
    logger.info(f"Selected chars ({len(selected_chars)}):\t{' '.join(sorted(selected_chars))}")
  if len(rest_chars) > 0:
    logger.info(f"Deselected chars ({len(rest_chars)}):\t{' '.join(sorted(rest_chars))}")

  if len(selected_word_counts) > 0:
    logger.info(
      f"Selected words count: {min(selected_word_counts)} (min), {np.mean(selected_word_counts):.0f} (mean), {np.median(selected_word_counts):.0f} (median), {max(selected_word_counts)} (max)")
  if len(deselected_word_counts) > 0:
    logger.info(
      f"Deselected words count: {min(deselected_word_counts)} (min), {np.mean(deselected_word_counts):.0f} (mean), {np.median(deselected_word_counts):.0f} (median), {max(deselected_word_counts)} (max)")

  if len(selected_durations_s) > 0:
    x = selected_durations_s
    logger.info(
      f"Selected utterance durations: {min(x):.0f}s (min), {np.mean(x):.0f}s (mean), {np.median(x):.0f}s (median), {max(x):.0f}s (max)")
  if len(deselected_durations_s) > 0:
    x = deselected_durations_s
    logger.info(
      f"Deselected utterance durations: {min(x):.0f}s (min), {np.mean(x):.0f}s (mean), {np.median(x):.0f}s (median), {max(x):.0f}s (max)")


def get_n_gram_stats_df(utterances: Utterances, selection: Selection, n: int) -> DataFrame:
  columns = [
    f"{n}-gram",
    "Selected #",
    "Not selected #",
    "Total #",
    "Selected %",
    "Not selected %",
    "Total %",
  ]

  if len(utterances) == 0:
    empty_stats_df = DataFrame(data=[], columns=columns)
    return empty_stats_df

  selected = [
    utterance_to_symbols(
      utterance,
      utterances.symbol_format,
      utterances.language
    )
    for utterance_id, utterance in tqdm(utterances.items()) if utterance_id in selection
  ]

  deselected = [
    utterance_to_symbols(
      utterance,
      utterances.symbol_format,
      utterances.language
    )
    for utterance_id, utterance in tqdm(utterances.items()) if utterance_id not in selection
  ]

  selected_n_grams = [n_gram for utterance_symbols in tqdm(selected)
                      for n_gram in get_ngrams(utterance_symbols, n)]
  deselected_n_grams = [n_gram for utterance_symbols in tqdm(deselected)
                        for n_gram in get_ngrams(utterance_symbols, n)]

  total_n_grams = selected_n_grams + deselected_n_grams
  total_n_grams_ordered = OrderedSet(list(sorted(total_n_grams)))
  all_n_grams_str = tuple(''.join(n_grams) for n_grams in total_n_grams_ordered)
  total_n_grams_count = __get_ngram_stats_count(total_n_grams, total_n_grams_ordered)
  total_n_grams_percent = __get_ngram_stats_percent(total_n_grams_count)
  selected_n_grams_count = __get_ngram_stats_count(selected_n_grams, total_n_grams_ordered)
  selected_n_grams_percent = __get_ngram_stats_percent(selected_n_grams_count)
  deselected_n_grams_count = __get_ngram_stats_count(deselected_n_grams, total_n_grams_ordered)
  deselected_n_grams_percent = __get_ngram_stats_percent(deselected_n_grams_count)

  utterances = np.array([
    all_n_grams_str,
    selected_n_grams_count,
    deselected_n_grams_count,
    total_n_grams_count,
    selected_n_grams_percent,
    deselected_n_grams_percent,
    total_n_grams_percent
  ]).T

  n_gram_stats_df = DataFrame(data=utterances, columns=columns)

  return n_gram_stats_df


def __get_ngram_stats_count(ngrams: List[Tuple[str, ...]], ngrams_order: OrderedSet[Tuple[str, ...]]) -> Tuple[int, ...]:
  counter = Counter(ngrams)
  res = tuple(counter[n_gram] if n_gram in counter else 0 for n_gram in ngrams_order)
  return res


def __get_ngram_stats_percent(ngram_stats_count: Tuple[int, ...]) -> Tuple[float, ...]:
  total_count = sum(ngram_stats_count)
  if total_count == 0:
    ngram_stats_percent = tuple(np.nan for _ in ngram_stats_count)
  else:
    ngram_stats_percent = tuple(count / total_count * 100 for count in ngram_stats_count)
  return ngram_stats_percent


def __get_utterances_word_counts(utterances: List[Utterance]) -> List[int]:
  word_counts = []
  for utterance in tqdm(utterances):
    words = utterance_to_text(utterance).split(" ")
    word_counts.append(len(words))
  return word_counts


def __get_utterances_durations(utterances: List[Utterance], avg_chars_per_s: float) -> List[float]:
  durations = [
    get_utterance_duration_s(utterance, avg_chars_per_s)
    for utterance in tqdm(utterances)
  ]
  return durations


def __get_utterance_duration_distribution(utterance_durations: List[float]) -> OrderedDictType[int, float]:
  duration_distribution: OrderedDictType[int, float] = OrderedDict()
  current_step_duration = 0

  # unsorted_utterances_rounded = [round(x, 1) for x in utterance_durations]
  unsorted_utterances_rounded = [round(x) for x in utterance_durations]
  while len(unsorted_utterances_rounded) > 0:
    to_remove = []
    for current_duration in unsorted_utterances_rounded:
      # if current_duration == round(current_step_duration, 1):
      if current_duration == round(current_step_duration):
        if current_step_duration not in duration_distribution:
          duration_distribution[current_step_duration] = 1
        else:
          duration_distribution[current_step_duration] += 1
        to_remove.append(current_duration)
    for item in to_remove:
      unsorted_utterances_rounded.remove(item)
    # current_step_duration += 0.1
    current_step_duration += 1
  return duration_distribution


def __log_distribution(distribution: OrderedDictType[int, float]) -> None:
  logger = getLogger(__name__)
  total_length = sum([step_duration * step_occurrences for step_duration,
                      step_occurrences in distribution.items()])
  # total_count = sum(distribution.values())
  current_summed_occurrences = 0
  current_summed_lengths = 0
  for step_duration, step_occurrences in distribution.items():
    current_length = step_duration * step_occurrences
    current_summed_lengths += current_length
    current_summed_occurrences += step_occurrences
    # logger.info(f"{step_duration:.1f}s: {step_occurrences} ({step_occurrences/total_count*100:.2f}%) ({current_summed_occurrences/total_count*100:.2f}%)")
    logger.info(f"{step_duration:.0f}s: {step_occurrences} ({current_length/total_length*100:.2f}%) ({current_summed_lengths/total_length*100:.2f}%)")
