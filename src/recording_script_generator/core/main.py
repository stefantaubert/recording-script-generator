import math
from collections import Counter, OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import enchant
from ordered_set import OrderedSet
from recording_script_generator.core.text_extraction import (
    contains_eng_proper_names, contains_undesired_text, file_to_utterances,
    get_minimum_frequency, get_non_dict_words_amount, is_sentence,
    strip_punctuation_words, words_contain_acronyms)
from recording_script_generator.utils import detect_ids_from_tex
from text_selection import greedy_kld_uniform_ngrams_iterations
from text_selection.greedy_export import (greedy_ngrams_durations_advanced,
                                          greedy_ngrams_epochs)
from text_selection.greedy_kld_export import \
    greedy_kld_uniform_ngrams_seconds_with_preselection
from text_selection.selection import SelectionMode
from text_selection.utils import DurationBoundary
from text_utils import EngToIPAMode, Language, SymbolFormat, Symbols
from text_utils import change_ipa as change_ipa_method
from text_utils import (clear_ipa_cache, symbols_to_ipa, text_normalize,
                        text_to_symbols)
from text_utils.text import change_symbols
from text_utils.types import Symbol
from tqdm import tqdm

SentenceId = int
ReadingPassage = Symbols
Representation = Symbols
ReadingPassages = Dict[int, ReadingPassage]
Representations = Dict[int, Representation]
SymbolPassages = Dict[int, Symbols]


class PreparationTarget(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1
  BOTH = 2


class Mode(IntEnum):
  SELECT = 0
  DESELECT = 1


@dataclass()
class PreparationData:
  language: Language
  reading_passages_format: SymbolFormat
  reading_passages: ReadingPassages
  representations_format: SymbolFormat
  representations: Representations
  selected: OrderedSet[SentenceId]


def get_sentences_from_text(text: str, lang: Language, text_format: SymbolFormat) -> Set[str]:
  utterances = file_to_utterances(text, lang, text_format)
  sentences = []

  for utterance in utterances:
    if not is_sentence(utterance, lang, text_format):
      continue
    sentences.append(utterance)
  return sentences


MAX_THREAD_COUNT = cpu_count() - 1
DEFAULT_CHUNK_SIZE_THREADS = 20
print(f"Using {MAX_THREAD_COUNT} threads with {DEFAULT_CHUNK_SIZE_THREADS} chunks...")


def read_textfile(path: Path) -> str:
  content = path.read_text()
  content = content.replace("\n", " ")
  return content


def add_corpus_from_text_files(files: List[Path], lang: Language, text_format: SymbolFormat) -> PreparationData:
  logger = getLogger(__name__)
  logger.info("Reading text files...")
  files=set(list(files)[:100])
  with ProcessPoolExecutor(max_workers=MAX_THREAD_COUNT) as ex:
    res = list(tqdm(ex.map(read_textfile, files, chunksize=DEFAULT_CHUNK_SIZE_THREADS), total=len(files)))
  logger.info("Done.")

  return add_corpus_from_texts(
    texts=res,
    lang=lang,
    text_format=text_format,
  )


def add_corpus_from_texts(texts: List[str], lang: Language, text_format: SymbolFormat) -> PreparationData:
  logger = getLogger(__name__)
  method = partial(get_sentences_from_text, lang=lang, text_format=text_format)

  with ProcessPoolExecutor(max_workers=MAX_THREAD_COUNT) as ex:
    res: List[Set[str]] = list(
      tqdm(ex.map(method, texts, chunksize=DEFAULT_CHUNK_SIZE_THREADS), total=len(texts)))

  reading_passages: Dict[SentenceId, ReadingPassage] = {}
  for text_sentences in res:
    for text_sentence in text_sentences:
      symbols = text_to_symbols(
        text=text_sentence,
        lang=lang,
        text_format=text_format,
      )

      utterance_id = len(reading_passages)
      reading_passages[utterance_id] = symbols

  #     total_utterance_count += len(utterances)

  # selected_percent = len(reading_passages) / total_utterance_count
  # logger.info(
  #   f"{selected_percent*100:.2f}% ({len(reading_passages)}) of all {total_utterance_count} utterances were sentences and thus selected.")

  result = PreparationData(
    reading_passages=reading_passages,
    language=lang,
    reading_passages_format=text_format,
    representations=reading_passages.copy(),
    representations_format=text_format,
    selected=OrderedSet(),
  )

  return result


def normalize(data: PreparationData, target: PreparationTarget) -> None:
  logger = getLogger(__name__)
  targets: List[Tuple[SymbolPassages, SymbolFormat]] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Normalizing reading passages...")
    targets.append((data.reading_passages, data.reading_passages_format))
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Normalizing representations...")
    targets.append((data.representations, data.representations_format))

  for target_data, target_format in targets:
    for utterance_id, symbols in tqdm(target_data.items()):
      normalized_text = text_normalize(
        text=''.join(symbols),
        lang=data.language,
        text_format=target_format,
      )

      symbols = text_to_symbols(
        text=normalized_text,
        lang=data.language,
        text_format=target_format,
      )

      target_data[utterance_id] = symbols


def convert_to_ipa(data: PreparationData, target: PreparationTarget, mode: Optional[EngToIPAMode]) -> None:
  logger = getLogger(__name__)
  targets: List[Tuple[SymbolPassages, SymbolFormat]] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Converting reading passages to IPA...")
    targets.append((data.reading_passages, data.reading_passages_format))
    data.reading_passages_format = SymbolFormat.PHONEMES_IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Converting representations to IPA...")
    targets.append((data.representations, data.representations_format))
    data.representations_format = SymbolFormat.PHONEMES_IPA

  for target_data, target_format in targets:
    for utterance_id, symbols in tqdm(target_data.items()):
      new_symbols, new_format = symbols_to_ipa(
        symbols=symbols,
        symbols_format=target_format,
        lang=data.language,
        mode=mode,
        consider_annotations=False,  # not useful
      )

      assert new_format == SymbolFormat.PHONEMES_IPA
      target_data[utterance_id] = new_symbols

  clear_ipa_cache()


def change_ipa(data: PreparationData, target: PreparationTarget, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> None:
  logger = getLogger(__name__)
  targets: List[SymbolPassages] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Changing reading passages...")
    targets.append(data.reading_passages)
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Changing representations...")
    targets.append(data.representations)

  for target_data in targets:
    for utterance_id, symbols in tqdm(target_data.items()):
      new_symbols = change_ipa_method(
        symbols=symbols,
        ignore_tones=ignore_tones,
        ignore_arcs=ignore_arcs,
        ignore_stress=ignore_stress,
        break_n_thongs=break_n_thongs,
        build_n_thongs=build_n_thongs,
        language=data.language,
      )

      target_data[utterance_id] = new_symbols


def change_text(data: PreparationData, target: PreparationTarget, remove_space_around_punctuation: bool) -> None:
  logger = getLogger(__name__)
  targets: List[SymbolPassages] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Changing reading passages...")
    targets.append(data.reading_passages)
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Changing representations...")
    targets.append(data.representations)

  for target_data in targets:
    for utterance_id, symbols in tqdm(target_data.items()):
      new_symbols = change_symbols(
        symbols=symbols,
        remove_space_around_punctuation=remove_space_around_punctuation,
        lang=data.language,
      )

      target_data[utterance_id] = new_symbols


def select_all_utterances(data: PreparationData):
  data.selected |= OrderedSet(data.reading_passages.keys())


def deselect_all_utterances(data: PreparationData):
  data.selected = OrderedSet()


def __remove_utterances(utterance_ids: Set[int], data: PreparationData) -> None:
  for remove_utterance_id in utterance_ids:
    data.reading_passages.pop(remove_utterance_id)
    data.representations.pop(remove_utterance_id)
    if remove_utterance_id in data.selected:
      data.selected.remove(remove_utterance_id)


def _remove_utterances_with_logging(utterance_ids: Set[int], data: PreparationData) -> None:
  logger = getLogger(__name__)
  old_len = len(data.representations)
  for utterance_id in utterance_ids:
    utterance_repr = ''.join(data.reading_passages[utterance_id])
    logger.info(f"Ignore \"{utterance_repr}\" ({utterance_id}).")
  __remove_utterances(utterance_ids, data)
  logger.info(
    f"Removed {len(utterance_ids)} of {old_len} utterances ({len(utterance_ids)/old_len*100:.2f}%) and obtained {len(data.representations)} utterances.")


def get_single_target(data: PreparationData, target: PreparationTarget) -> SymbolPassages:
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH:
    logger.error("Target BOTH is not supported in this case!")
    raise Exception()
  if target == PreparationTarget.READING_PASSAGES:
    return data.reading_passages
  assert target == PreparationTarget.REPRESENTATIONS
  return data.representations


def remove_deselected(data: PreparationData) -> None:
  remove = OrderedSet(data.reading_passages.keys()) - data.selected

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_undesired_text(data: PreparationData, target: PreparationTarget, undesired: Set[str]) -> None:
  remove = OrderedSet()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    if contains_undesired_text(utterance, undesired=undesired, ignore_case=True):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_duplicate_utterances(data: PreparationData, target: PreparationTarget) -> None:
  remove = OrderedSet()
  already_exist: Set[Tuple[str, ...]] = set()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance_symbols_tuple = tuple(utterance_symbols)
    if utterance_symbols_tuple in already_exist:
      remove.add(utterance_id)
    else:
      already_exist.add(utterance_symbols_tuple)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_proper_names(data: PreparationData, target: PreparationTarget) -> None:
  remove = OrderedSet()
  if data.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)

    if contains_eng_proper_names(utterance):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_acronyms(data: PreparationData, target: PreparationTarget) -> None:
  remove = OrderedSet()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    if words_contain_acronyms(words_non_punctuation):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_undesired_sentence_lengths(data: PreparationData, target: PreparationTarget, min_word_count: Optional[int], max_word_count: Optional[int]) -> None:
  remove = OrderedSet()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_count = len(words)

    if min_word_count is not None and words_count < min_word_count:
      remove.add(utterance_id)
    elif max_word_count is not None and words_count > max_word_count:
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_unknown_words(data: PreparationData, target: PreparationTarget, max_unknown_word_count: int) -> None:
  if data.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  lexicon = enchant.Dict("en_US")
  remove = OrderedSet()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, lexicon)
    if non_dict_words_amount > max_unknown_word_count:
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_too_seldom_words(data: PreparationData, target: PreparationTarget, min_occurrence_count: int) -> None:
  remove = OrderedSet()
  stripped_words: Dict[int, List[str]] = {}
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)
    stripped_words[utterance_id] = words_non_punctuation

  words_counter = Counter(word.lower() for words in stripped_words.values()
                          for word in words)

  for utterance_id, words in stripped_words.items():
    min_freq = get_minimum_frequency(words, words_counter)

    if min_freq < min_occurrence_count:
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def merge(data: List[PreparationData]) -> PreparationData:
  assert len(data) > 0

  first_entry = data[0]

  merged_data = PreparationData(
    reading_passages=dict(),
    language=first_entry.language,
    reading_passages_format=first_entry.reading_passages_format,
    representations=dict(),
    representations_format=first_entry.representations_format,
    selected=OrderedSet(),
  )

  id_counter = 0
  for data_to_merge in data:
    assert data_to_merge.language == merged_data.language
    assert data_to_merge.reading_passages_format == merged_data.reading_passages_format
    assert data_to_merge.representations_format == merged_data.representations_format
    sentence_id_mapping = dict()
    for sentence_id, reading_passage_symbols in data_to_merge.reading_passages.items():
      representation_symbols = data_to_merge.representations[sentence_id]
      merged_data.reading_passages[id_counter] = reading_passage_symbols
      merged_data.representations[id_counter] = representation_symbols
      sentence_id_mapping[sentence_id] = id_counter
      id_counter += 1

    for selected_sentence_id in data_to_merge.selected:
      new_sentence_id = sentence_id_mapping[selected_sentence_id]
      merged_data.selected.add(new_sentence_id)

  return merged_data


def select_kld_ngrams_iterations(data: PreparationData, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]):
  logger = getLogger(__name__)
  rest = OrderedDict({k: v for k, v in data.representations.items() if k not in data.selected})
  new_selected = greedy_kld_uniform_ngrams_iterations(
    data=rest,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    iterations=iterations,
  )

  data.selected |= new_selected

  logger.info(f"Added {len(new_selected)} utterances to selection.")


def select_kld_ngrams_duration(data: PreparationData, n_gram: int, minutes: float, reading_speed_chars_per_s: float, ignore_symbols: Set[Symbol], boundary: DurationBoundary):
  logger = getLogger(__name__)
  selected_representations = OrderedDict(
    {k: v for k, v in data.representations.items() if k in data.selected})

  logger.info(f"Already selected: {len(selected_representations)}.")

  non_selected_durations = {k: len(v) / reading_speed_chars_per_s
                            for k, v in data.reading_passages.items()
                            if k not in data.selected}

  non_selected_reading_passages = OrderedDict(
    {k: v for k, v in data.reading_passages.items() if k not in data.selected})
  non_selected_representations = OrderedDict(
    {k: v for k, v in data.representations.items() if k not in data.selected})

  new_selected = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=non_selected_representations,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=minutes * 60,
    durations_s=non_selected_durations,
    preselection=selected_representations,
    duration_boundary=boundary,
    mp=True,
  )

  data.selected |= new_selected
  selected_duration = sum(len(v) / reading_speed_chars_per_s / 60
                          for k, v in non_selected_reading_passages.items() if k in new_selected)
  logger.info(f"Added {len(new_selected)} utterances to selection ({selected_duration:.2f}min).")


# def filter_after_duration(corpus: Dict[int, float], min_duration_incl: float, max_duration_excl: float) -> Set[int]:
#   assert min_duration_incl >= 0
#   assert max_duration_excl >= 0

#   filtered_utterance_indicies = set()

#   for utterance_id, utterance_duration in corpus.items():
#     if min_duration_incl <= utterance_duration < max_duration_excl:
#       filtered_utterance_indicies.add(utterance_id)

#   return filtered_utterance_indicies


def select_greedy_ngrams_epochs(data: PreparationData, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]]):
  logger = getLogger(__name__)
  rest = OrderedDict({k: v for k, v in data.representations.items() if k not in data.selected})
  new_selected = greedy_ngrams_epochs(
    data=rest,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )

  data.selected |= new_selected

  logger.info(f"Added {len(new_selected)} utterances to selection.")


def select_greedy_ngrams_duration(data: PreparationData, n_gram: int, minutes: float, reading_speed_chars_per_s: float, ignore_symbols: Optional[Set[Symbol]], mode: SelectionMode):
  logger = getLogger(__name__)
  non_selected_reading_passages = OrderedDict(
    {k: v for k, v in data.reading_passages.items() if k not in data.selected})
  non_selected_representations = OrderedDict(
    {k: v for k, v in data.representations.items() if k not in data.selected})
  durations = {k: len(v) / reading_speed_chars_per_s for k,
               v in non_selected_reading_passages.items()}
  new_selected = greedy_ngrams_durations_advanced(
    data=non_selected_representations,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    target_duration=minutes * 60,
    durations=durations,
    mode=mode,
  )

  data.selected |= new_selected
  selected_duration = sum(len(v) / reading_speed_chars_per_s / 60
                          for k, v in non_selected_reading_passages.items() if k in new_selected)
  logger.info(f"Added {len(new_selected)} utterances to selection ({selected_duration:.2f}min).")


def select_from_tex(data: PreparationData, tex: str) -> None:
  logger = getLogger(__name__)
  ids_in_tex = detect_ids_from_tex(tex)

  new_ids = ids_in_tex - data.selected
  if len(new_ids) > 0:
    logger.error("Added new entries:")
    logger.info(new_ids)
    raise Exception()

  final_ids = OrderedSet()
  for current_id in data.selected:
    if current_id in ids_in_tex:
      final_ids.add(current_id)

  removed_count = len(data.selected - ids_in_tex)

  if removed_count == 0:
    logger.info("Nothing to do.")
  else:
    old_len = len(data.selected)
    remove_ids = data.selected - ids_in_tex
    data.selected -= remove_ids
    if old_len == 0:
      logger.info(
        f"Removed {len(remove_ids)} sentences from selection (100%).")
    else:
      # ids = ','.join(list(map(str, list(sorted(remove_ids)))))
      logger.info(
        f"Removed {len(remove_ids)} sentences from selection ({len(remove_ids)/old_len*100:.2}%).")
    for removed_id in sorted(remove_ids):
      logger.info(f"- {removed_id}: {''.join(data.reading_passages[removed_id])}")
