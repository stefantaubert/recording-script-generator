import logging
import math
import string
from collections import Counter, OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from logging import getLogger
from multiprocessing import Manager, cpu_count
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, Optional, Set, Tuple,
                    cast, overload)

import enchant
from ordered_set import OrderedSet
from recording_script_generator.core.text_extraction import (
    contains_eng_proper_names, contains_undesired_text, file_to_utterances,
    get_minimum_frequency, get_non_dict_words_amount, is_sentence,
    strip_punctuation_words, words_contain_acronyms)
from recording_script_generator.utils import detect_ids_from_tex
from sentence2pronunciation import prepare_cache_mp
from sentence2pronunciation.core import sentences2pronunciations_from_cache_mp
from text_selection import greedy_kld_uniform_ngrams_iterations
from text_selection.greedy_export import (greedy_ngrams_durations_advanced,
                                          greedy_ngrams_epochs)
from text_selection.greedy_kld_export import \
    greedy_kld_uniform_ngrams_seconds_with_preselection
from text_selection.selection import SelectionMode
from text_selection.utils import DurationBoundary
from text_utils import EngToIPAMode, Language, Symbol, SymbolFormat, Symbols
from text_utils import change_ipa as change_ipa_method
from text_utils import (change_symbols, prepare_symbols_to_ipa, symbols_join,
                        symbols_map_arpa_to_ipa, symbols_to_ipa,
                        text_normalize, text_to_symbols)
from text_utils.pronunciation.G2p_cache import get_eng_g2p
from text_utils.pronunciation.main import get_eng_to_arpa_lookup_method
from text_utils.pronunciation.pronunciation_dict_cache import \
    get_eng_pronunciation_dict_arpa
from tqdm import tqdm

SentenceId = int
# ReadingPassage = Symbols
# Representation = Symbols
#ReadingPassages = Dict[int, ReadingPassage]
#Representations = Dict[int, Representation]
SymbolPassages = Dict[int, Symbols]


class Mode(IntEnum):
  SELECT = 0
  DESELECT = 1


# @dataclass()
# class PreparationData:
#   language: Language
#   reading_passages_format: SymbolFormat
#   reading_passages: ReadingPassages
#   representations_format: SymbolFormat
#   representations: Representations
#   selected: OrderedSet[SentenceId]

@dataclass()
class Metadata():
  language: Language
  selected: OrderedSet[SentenceId]


class Passages(Dict[int, Symbols]):
  symbol_format: SymbolFormat


class ReadingPassages(Passages):
  pass


class Representations(Passages):
  pass


DEFAULT_N_JOBS = cpu_count() - 1
DEFAULT_CHUNK_SIZE_PROCESSES = 1000
print(f"Using {DEFAULT_N_JOBS} threads with {DEFAULT_CHUNK_SIZE_PROCESSES} chunks...")


def read_textfile(path: Path) -> str:
  content = path.read_text()
  content = content.replace("\n", " ")
  return content


def add_corpus_from_text_files(files: Set[Path], lang: Language, text_format: SymbolFormat) -> Tuple[ReadingPassages, Representations]:
  logger = getLogger(__name__)
  logger.info("Reading text files...")
  #files = set(list(files)[:5])
  with ThreadPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res = list(tqdm(ex.map(read_textfile, files), total=len(files)))
  logger.info("Done.")

  return add_corpus_from_texts(
    texts=res,
    lang=lang,
    text_format=text_format,
  )


def get_sentences_from_text(text: str, lang: Language, text_format: SymbolFormat) -> Set[str]:
  utterances = file_to_utterances(text, lang, text_format)
  sentences = []

  for utterance in utterances:
    if not is_sentence(utterance, lang, text_format):
      continue
    sentences.append(utterance)
  return sentences


def add_corpus_from_texts(texts: List[str], lang: Language, text_format: SymbolFormat) -> Tuple[Metadata, ReadingPassages, Representations]:
  logger = getLogger(__name__)
  method = partial(get_sentences_from_text, lang=lang, text_format=text_format)
  logger.info("Detecting valid sentences...")
  tqdm_steps = 4
  chunksize = max(round(len(texts) / DEFAULT_N_JOBS / tqdm_steps), 1)
  logger.info(f"Assigning {chunksize} files to each processor core.")
  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res: List[Set[str]] = list(
      tqdm(ex.map(method, texts, chunksize=chunksize), total=len(texts)))

  passages: Dict[SentenceId, Symbols] = {}
  for text_sentences in res:
    for text_sentence in text_sentences:
      symbols = text_to_symbols(
        text=text_sentence,
        lang=lang,
        text_format=text_format,
      )

      utterance_id = len(passages)
      passages[utterance_id] = symbols

  #     total_utterance_count += len(utterances)

  # selected_percent = len(reading_passages) / total_utterance_count
  # logger.info(
  #   f"{selected_percent*100:.2f}% ({len(reading_passages)}) of all {total_utterance_count} utterances were sentences and thus selected.")
  logger.info("Done.")

  reading_passages = ReadingPassages(passages)
  reading_passages.symbol_format = text_format

  representations = Representations(passages)
  representations.symbol_format = text_format

  meta = Metadata(
    language=lang,
    selected=OrderedSet(),
  )

  return meta, reading_passages, representations


def return_input_too(inp: Any, method: Callable[[Any], Any]) -> Tuple[Any, Any]:
  return inp, method(inp)


def normalize_func(utterance_id_symbols_tuple: Tuple[int, Symbols], lang: Language, text_format: SymbolFormat) -> Tuple[int, Symbols]:
  utterance_id, symbols = utterance_id_symbols_tuple
  symbols_str = ''.join(symbols)
  result = text_normalize(
    text=symbols_str,
    text_format=text_format,
    lang=lang,
  )

  sentences = text_to_symbols(
    text=result,
    lang=lang,
    text_format=text_format,
  )

  return utterance_id, sentences


def normalize(data: Passages, metadata: Metadata) -> None:
  logger = getLogger(__name__)
  # maybe check if both data is same and then only do one and assign to other
  logger.info("Normalizing...")

  method = partial(normalize_func, lang=metadata.language, text_format=data.symbol_format)

  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res: SymbolPassages = dict(
      tqdm(ex.map(method, data.entries.items(), chunksize=DEFAULT_CHUNK_SIZE_PROCESSES), total=len(data.entries)))
  logger.info("Updating existing data...")
  data.entries.update(res)
  logger.info("Done.")


# def convert_to_ipa_func(utterance_id_symbols_tuple: Tuple[int, Symbols], lang: Language, text_format: SymbolFormat, mode: Optional[EngToIPAMode]) -> Tuple[int, Symbols]:
#   utterance_id, symbols = utterance_id_symbols_tuple
#   new_symbols, new_format = symbols_to_ipa(
#     symbols=symbols,
#     symbols_format=text_format,
#     lang=lang,
#     mode=mode,
#     consider_annotations=False,  # as it is not useful
#   )

#   assert new_format == SymbolFormat.PHONEMES_IPA
#   return utterance_id, new_symbols

def convert_eng_passages_to_arpa(data: SymbolPassages) -> None:
  logger = getLogger(__name__)
  logger.info("Loading dictionaries...")
  get_eng_g2p()
  get_eng_pronunciation_dict_arpa()
  logger.info("Done.")

  logger.info("Preparing conversion...")
  sentences = set(data.values())
  cache = prepare_cache_mp(
    sentences=sentences,
    annotation_split_symbol=None,
    chunksize=10000,
    consider_annotation=False,
    get_pronunciation=get_eng_to_arpa_lookup_method(),
    ignore_case=True,
    n_jobs=DEFAULT_N_JOBS,
    split_on_hyphen=True,
    trim_symbols=set(string.punctuation),
  )
  logger.info(f"Done. Retrieved {len(cache)} unique words (incl. punctuation).")

  logger.info("Converting to ARPA...")
  sentence_pronunciations = sentences2pronunciations_from_cache_mp(
    sentences=sentences,
    cache=cache,
    annotation_split_symbol=None,
    chunksize=10000,
    consider_annotation=False,
    ignore_case=True,
    n_jobs=DEFAULT_N_JOBS,
  )
  logger.info("Done.")

  # Inplace to reduce memory
  logger.info("Updating existing data...")
  for sentence_id, old_pronunciation in data.items():
    data[sentence_id] = sentence_pronunciations[old_pronunciation]
  logger.info("Done.")


def convert_eng_to_arpa(metadata: Metadata, data: Passages) -> None:
  logger = getLogger(__name__)
  targets: List[Tuple[SymbolPassages, SymbolFormat]] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    targets.append((data.reading_passages, data.reading_passages_format))
    data.reading_passages_format = SymbolFormat.PHONEMES_IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    targets.append((data.representations, data.representations_format))
    data.representations_format = SymbolFormat.PHONEMES_IPA

  prn_logger = getLogger("text_utils.pronunciation.main")
  prn_logger.setLevel(logging.WARNING)
  for target_data, target_format in targets:
    if target_format == PreparationTarget.READING_PASSAGES:
      logger.info("Converting reading passages to ARPA...")
    elif target_format == PreparationTarget.REPRESENTATIONS:
      logger.info("Converting representations to ARPA...")
    convert_eng_passages_to_arpa(target_data)


def map_to_tup_ipa(tup: Tuple[int, Symbols]) -> Tuple[int, Symbols]:
  utterance_id, arpa_symbols = tup
  ipa_symbols = symbols_map_arpa_to_ipa(
    arpa_symbols=arpa_symbols,
    ignore={},
    replace_unknown=False,
    replace_unknown_with=None,
  )
  return utterance_id, ipa_symbols


def map_passages_to_ipa(passages: SymbolPassages):
  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res: SymbolPassages = dict(
      tqdm(ex.map(map_to_tup_ipa, passages.items(),
           chunksize=DEFAULT_CHUNK_SIZE_PROCESSES), total=len(passages))
    )
  passages.update(res)


def map_to_ipa(metadata: Metadata, data: Passages) -> None:
  logger = getLogger(__name__)
  targets: List[Tuple[SymbolPassages, SymbolFormat]] = []
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    targets.append((data.reading_passages, data.reading_passages_format))
    data.reading_passages_format = SymbolFormat.PHONEMES_IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    targets.append((data.representations, data.representations_format))
    data.representations_format = SymbolFormat.PHONEMES_IPA

  for target_data, target_format in targets:
    if target_format == PreparationTarget.READING_PASSAGES:
      logger.info("Mapping reading passages to IPA...")
    elif target_format == PreparationTarget.REPRESENTATIONS:
      logger.info("Converting representations to IPA...")
    map_passages_to_ipa(target_data)


def change_ipa(metadata: Metadata, data: Passages, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> None:
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


def change_text(metadata: Metadata, data: Passages, remove_space_around_punctuation: bool) -> None:
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


def select_all_utterances(metadata: Metadata, data: Passages):
  data.selected |= OrderedSet(data.reading_passages.keys())


def deselect_all_utterances(metadata: Metadata, data: Passages):
  data.selected = OrderedSet()


def __remove_utterances(utterance_ids: Set[int], metadata: Metadata, data: Passages) -> None:
  for remove_utterance_id in utterance_ids:
    data.reading_passages.pop(remove_utterance_id)
    data.representations.pop(remove_utterance_id)
    if remove_utterance_id in data.selected:
      data.selected.remove(remove_utterance_id)


def _remove_utterances_with_logging(utterance_ids: Set[int], metadata: Metadata, data: Passages) -> None:
  logger = getLogger(__name__)
  old_len = len(data.representations)
  for utterance_id in utterance_ids:
    utterance_repr = ''.join(data.reading_passages[utterance_id])
    logger.info(f"Ignore \"{utterance_repr}\" ({utterance_id}).")
  __remove_utterances(utterance_ids, data)
  logger.info(
    f"Removed {len(utterance_ids)} of {old_len} utterances ({len(utterance_ids)/old_len*100:.2f}%) and obtained {len(data.representations)} utterances.")


def get_single_target(metadata: Metadata, data: Passages) -> SymbolPassages:
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH:
    logger.error("Target BOTH is not supported in this case!")
    raise Exception()
  if target == PreparationTarget.READING_PASSAGES:
    return data.reading_passages
  assert target == PreparationTarget.REPRESENTATIONS
  return data.representations


def remove_deselected(metadata: Metadata, data: Passages) -> None:
  remove = OrderedSet(data.reading_passages.keys()) - data.selected

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_undesired_text(metadata: Metadata, data: Passages, undesired: Set[str]) -> None:
  remove = OrderedSet()
  target_data = get_single_target(data, target)
  for utterance_id, utterance_symbols in target_data.items():
    utterance = ''.join(utterance_symbols)
    if contains_undesired_text(utterance, undesired=undesired, ignore_case=True):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_duplicate_utterances(metadata: Metadata, data: Passages) -> None:
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


def remove_utterances_with_proper_names(metadata: Metadata, data: Passages) -> None:
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


def check_utterance_contain_acronyms(utterance_tuple: Tuple[int, Symbols]) -> Tuple[int, bool]:
  utterance_id, utterance = utterance_tuple
  utterance = ''.join(utterance)
  words = utterance.split(" ")
  words_non_punctuation = strip_punctuation_words(words)

  result = words_contain_acronyms(words_non_punctuation)
  return utterance_id, result


def remove_utterances_with_acronyms(metadata: Metadata, data: Passages) -> None:
  remove = OrderedSet()
  logger = getLogger(__name__)
  tqdm_steps = 4
  chunksize = min(round(len(data) / DEFAULT_N_JOBS / tqdm_steps), 100)
  chunksize = 1
  logger.info(f"Assigning {chunksize} utterances to each processor core.")


  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res: Dict[int, bool] = dict(
        tqdm(ex.map(check_utterance_contain_acronyms, data.items(), chunksize=chunksize), total=len(data)))

  for utterance_id, dont_include in tqdm(res.items()):
    if dont_include:
      remove.add(data[utterance_id])

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_undesired_sentence_lengths(metadata: Metadata, data: Passages, min_word_count: Optional[int], max_word_count: Optional[int]) -> None:
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


def remove_utterances_with_unknown_words(metadata: Metadata, data: Passages, max_unknown_word_count: int) -> None:
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


def remove_utterances_with_too_seldom_words(metadata: Metadata, data: Passages, min_occurrence_count: int) -> None:
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


def merge(data: List[Tuple[Metadata, ReadingPassages, Representations]]) -> Tuple[Metadata, ReadingPassages, Representations]:
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


def select_kld_ngrams_iterations(metadata: Metadata, data: Passages, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]):
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


def select_kld_ngrams_duration(metadata: Metadata, data: Passages, n_gram: int, minutes: float, reading_speed_chars_per_s: float, ignore_symbols: Set[Symbol], boundary: DurationBoundary):
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


def select_greedy_ngrams_epochs(metadata: Metadata, data: Passages, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]]):
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


def select_greedy_ngrams_duration(metadata: Metadata, data: Passages, n_gram: int, minutes: float, reading_speed_chars_per_s: float, ignore_symbols: Optional[Set[Symbol]], mode: SelectionMode):
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


def select_from_tex(metadata: Metadata, data: Passages, tex: str) -> None:
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
