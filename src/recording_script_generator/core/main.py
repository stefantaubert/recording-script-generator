import logging
import string
from collections import Counter, OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from enum import IntEnum
from functools import partial
from itertools import islice, zip_longest
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from time import perf_counter
from timeit import timeit
from typing import Any, Callable, Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

import enchant
import numpy as np
from ordered_set import OrderedSet
from recording_script_generator.core.text_extraction import (
    contains_eng_proper_names, contains_undesired_text, file_to_utterances,
    get_minimum_frequency, get_non_dict_words_amount, is_sentence,
    strip_punctuation_words, words_contain_acronyms)
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   UtteranceId, Utterances)
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
from text_utils import Language, Symbol, SymbolFormat, Symbols
from text_utils import change_ipa as change_ipa_method
from text_utils import (change_symbols, symbols_map_arpa_to_ipa,
                        text_normalize, text_to_symbols)
from text_utils.pronunciation.G2p_cache import get_eng_g2p
from text_utils.pronunciation.main import get_eng_to_arpa_lookup_method
from text_utils.pronunciation.pronunciation_dict_cache import \
    get_eng_pronunciation_dict_arpa
from tqdm import tqdm

DEFAULT_N_JOBS = cpu_count() - 1
DEFAULT_CHUNK_SIZE_PROCESSES = 1000
print(f"Using {DEFAULT_N_JOBS} threads with {DEFAULT_CHUNK_SIZE_PROCESSES} chunks...")


def read_textfile(path: Path) -> str:
  content = path.read_text()
  content = content.replace("\n", " ")
  return content


def add_corpus_from_text_files(files: Set[Path], lang: Language, text_format: SymbolFormat, limit: Optional[int]) -> Tuple[Selection, ReadingPassages, Representations]:
  logger = getLogger(__name__)
  logger.info("Reading text files...")
  if limit is not None:
    files = set(list(files)[:limit])
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


def add_corpus_from_texts(texts: List[str], lang: Language, text_format: SymbolFormat) -> Tuple[Selection, ReadingPassages, Representations]:
  logger = getLogger(__name__)
  method = partial(get_sentences_from_text, lang=lang, text_format=text_format)
  logger.info("Detecting valid sentences...")
  tqdm_steps = 4
  chunksize = max(round(len(texts) / DEFAULT_N_JOBS / tqdm_steps), 1)
  logger.info(f"Assigning {chunksize} files to each processor core.")
  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res: List[Set[str]] = list(
      tqdm(ex.map(method, texts, chunksize=chunksize), total=len(texts)))

  passages: Dict[UtteranceId, Symbols] = {}
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
  reading_passages.language = lang

  representations = Representations(passages)
  representations.symbol_format = text_format
  representations.language = lang

  selection = Selection()

  return selection, reading_passages, representations


def remove_non_existent_utterances(utterances: Utterances, target: Utterances) -> None:
  utterance_ids_to_remove = set(target.keys()) - set(utterances.keys())
  __remove_utterances_with_logging(utterance_ids_to_remove, target)


def return_input_too(inp: Any, method: Callable[[Any], Any]) -> Tuple[Any, Any]:
  return inp, method(inp)


def normalize_func(utterance_id_symbols_tuple: Tuple[UtteranceId, Symbols], lang: Language, text_format: SymbolFormat) -> Tuple[UtteranceId, Symbols]:
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


def normalize(utterances: Utterances, selection: Selection) -> None:
  logger = getLogger(__name__)
  # maybe check if both data is same and then only do one and assign to other
  logger.info("Normalizing...")

  method = partial(normalize_func, lang=utterances.language, text_format=utterances.symbol_format)

  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res = Utterances(
      tqdm(ex.map(method, utterances.items(), chunksize=DEFAULT_CHUNK_SIZE_PROCESSES), total=len(utterances)))
  logger.info("Updating existing utterances...")
  utterances.update(res)
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

def convert_eng_passages_to_arpa(utterances: Utterances, selection: Selection) -> None:
  logger = getLogger(__name__)
  logger.info("Loading dictionaries...")
  get_eng_g2p()
  get_eng_pronunciation_dict_arpa()
  logger.info("Done.")

  logger.info("Preparing conversion...")

  prn_logger = getLogger("text_utils.pronunciation.main")
  prn_logger.setLevel(logging.WARNING)

  sentences = set(utterances.values())
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

  # In-place to reduce memory
  logger.info("Updating existing utterances...")
  for utterance_id, old_pronunciation in utterances.items():
    utterances[utterance_id] = sentence_pronunciations[old_pronunciation]
  utterances.symbol_format = SymbolFormat.PHONEMES_ARPA
  logger.info("Done.")


def map_to_tup_ipa(tup: Tuple[int, Symbols]) -> Tuple[int, Symbols]:
  utterance_id, arpa_symbols = tup
  ipa_symbols = symbols_map_arpa_to_ipa(
    arpa_symbols=arpa_symbols,
    ignore={},
    replace_unknown=False,
    replace_unknown_with=None,
  )
  return utterance_id, ipa_symbols


def map_passages_to_ipa(passages: Utterances, selection: Selection) -> None:
  with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
    res = Utterances(
      tqdm(ex.map(map_to_tup_ipa, passages.items(),
           chunksize=DEFAULT_CHUNK_SIZE_PROCESSES), total=len(passages))
    )
  passages.update(res)
  passages.symbol_format = SymbolFormat.PHONEMES_IPA


def change_ipa(utterances: Utterances, selection: Selection, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> None:
  for utterance_id, symbols in tqdm(utterances.items()):
    new_symbols = change_ipa_method(
      symbols=symbols,
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      ignore_stress=ignore_stress,
      break_n_thongs=break_n_thongs,
      build_n_thongs=build_n_thongs,
      language=utterances.language,
    )

    utterances[utterance_id] = new_symbols


def change_text(utterances: Utterances, selection: Selection, remove_space_around_punctuation: bool) -> None:
  for utterance_id, symbols in tqdm(utterances.items()):
    new_symbols = change_symbols(
      symbols=symbols,
      remove_space_around_punctuation=remove_space_around_punctuation,
      lang=utterances.language,
    )

    utterances[utterance_id] = new_symbols


def select_all_utterances(utterances: Utterances, selection: Selection):
  selection |= OrderedSet(utterances.keys())


def deselect_all_utterances(utterances: Utterances, selection: Selection):
  selection.clear()


def remove_deselected(utterances: Utterances, selection: Selection) -> None:
  unselected_utterance_ids = set(utterances.keys()) - selection
  __remove_utterances_and_selection_with_logging(unselected_utterance_ids, utterances, selection)


def __remove_utterances_and_selection_with_logging(utterance_ids: Set[UtteranceId], utterances: Utterances, selection: Selection) -> None:
  __remove_utterances_from_selection_with_logging(utterance_ids, selection)
  __remove_utterances_with_logging(utterance_ids, utterances)


def __remove_utterances_with_logging(utterance_ids: Set[UtteranceId], utterances: Utterances) -> None:
  logger = getLogger(__name__)
  if len(utterances) == 0 or len(utterance_ids) == 0:
    logger.info("Nothing to remove.")
    return

  assert len(utterances) > 0
  old_count = len(utterances)
  log_count = 10
  for utterance_id in list(utterance_ids)[:log_count]:
    utterance_str = ''.join(utterances[utterance_id])
    logger.info(f"Removing \"{utterance_str}\" ({utterance_id})...")

  if len(utterance_ids) > log_count:
    logger.info(f"Removing {len(utterance_ids) - log_count} further utterance(s)...")

  for utterance_id in utterance_ids:
    utterances.pop(utterance_id)

  new_count = len(utterances)

  logger.info(
      f"Removed {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained {new_count} utterances.")


def __remove_utterances_from_selection_with_logging(utterance_ids: Set[UtteranceId], selection: Selection) -> None:
  logger = getLogger(__name__)
  if len(selection) == 0 or len(utterance_ids) == 0:
    logger.info("Nothing to deselect.")
    return

  old_count = len(selection)

  selection -= utterance_ids

  new_count = len(selection)

  logger.info(
      f"Deselected {old_count - new_count} of {old_count} utterances ({(old_count - new_count)/old_count*100:.2f}%) and obtained a selection of {new_count} utterances.")


def remove_utterances_with_undesired_text(utterances: Utterances, selection: Selection, undesired: Set[str]) -> None:
  remove = OrderedSet()
  for utterance_id, utterance_symbols in utterances.items():
    utterance = ''.join(utterance_symbols)
    if contains_undesired_text(utterance, undesired=undesired, ignore_case=True):
      remove.add(utterance_id)

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def remove_duplicate_utterances(utterances: Utterances, selection: Selection) -> None:
  remove = OrderedSet()
  already_exist: Set[Symbols] = set()
  for utterance_id, utterance_symbols in utterances.items():
    if utterance_symbols in already_exist:
      remove.add(utterance_id)
    else:
      already_exist.add(utterance_symbols)

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def remove_utterances_with_proper_names(utterances: Utterances, selection: Selection) -> None:
  if utterances.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  remove = OrderedSet()
  for utterance_id, utterance_symbols in utterances.items():
    utterance = ''.join(utterance_symbols)

    if contains_eng_proper_names(utterance):
      remove.add(utterance_id)

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def __check_utterance_contain_acronyms(utterance_tuple: Tuple[int, Symbols]) -> Tuple[int, bool]:
  utterance_id, utterance = utterance_tuple
  utterance = ''.join(utterance)
  words = utterance.split(" ")
  words_non_punctuation = strip_punctuation_words(words)

  result = words_contain_acronyms(words_non_punctuation)
  return utterance_id, result


def get_chunked_dict_keys(dictionary: OrderedDictType[Any, Any], chunk_size: int) -> OrderedSet[Any]:
  logger = getLogger(__name__)
  logger.info("Creating chunks...")
  now = perf_counter()
  keys = OrderedSet(dictionary.keys())
  #key_chunks = [keys[i:i + chunk_size] for i in tqdm(range(0, len(keys), chunk_size))]
  iterator = iter(keys)
  result = list(iter(lambda: list(islice(iterator, chunk_size)), list()))
  duration = perf_counter() - now
  logger.info(f"Done. Duration: {duration:.2f}s.")
  return result

  # return key_chunks


def get_chunk(utterances: Utterances, chunked_keys: OrderedSet[UtteranceId]) -> Utterances:
  logger = getLogger(__name__)
  now = perf_counter()
  logger.info("Getting chunk...")
  result = Utterances({k: utterances[k] for k in chunked_keys})
  result.language = utterances.language
  result.symbol_format = utterances.symbol_format
  duration = perf_counter() - now
  logger.info(f"Done. Duration: {duration:.2f}s.")
  return result


def remove_utterances_with_acronyms(utterances: Utterances, selection: Selection) -> None:
  logger = getLogger(__name__)
  outer_chunk_size = 5000000
  mp_steps = 3
  chunksize = round(outer_chunk_size / DEFAULT_N_JOBS / mp_steps)
  logger.info(f"Assigning {chunksize} utterances to {DEFAULT_N_JOBS} processor cores.")
  remove: OrderedSet[UtteranceId] = OrderedSet()
  logger.info("Chunking utterances...")
  chunked_keys = get_chunked_dict_keys(utterances, chunk_size=outer_chunk_size)
  logger.info("Done.")

  for chunk_nr, keys_chunk in tqdm(enumerate(chunked_keys, start=1)):
    logger.info(f"Running chunk {chunk_nr} of {len(chunked_keys)}...")
    chunk_utterances = get_chunk(utterances, keys_chunk)
    with ProcessPoolExecutor(max_workers=DEFAULT_N_JOBS) as ex:
      res: Dict[int, bool] = dict(
          tqdm(ex.map(__check_utterance_contain_acronyms, chunk_utterances.items(), chunksize=chunksize), total=len(chunk_utterances)))

    chunk_remove: OrderedSet[UtteranceId] = OrderedSet(
      [utterance_id for utterance_id, dont_include in res.items() if dont_include]
    )
    remove |= chunk_remove
    logger.info(f"Done with chunk {chunk_nr}.")
  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def remove_utterances_with_undesired_sentence_lengths(utterances: Utterances, selection: Selection, min_word_count: Optional[int], max_word_count: Optional[int]) -> None:
  remove = OrderedSet()
  for utterance_id, utterance_symbols in utterances.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_count = len(words)

    if min_word_count is not None and words_count < min_word_count:
      remove.add(utterance_id)
    elif max_word_count is not None and words_count > max_word_count:
      remove.add(utterance_id)

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def remove_utterances_with_unknown_words(utterances: Utterances, selection: Selection, max_unknown_word_count: int) -> None:
  if utterances.language != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    raise Exception()

  lexicon = enchant.Dict("en_US")
  remove = OrderedSet()
  for utterance_id, utterance_symbols in utterances.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, lexicon)
    if non_dict_words_amount > max_unknown_word_count:
      remove.add(utterance_id)

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def remove_utterances_with_too_seldom_words(utterances: Utterances, selection: Selection, min_occurrence_count: int) -> None:
  remove = OrderedSet()
  stripped_words: Dict[int, List[str]] = {}
  for utterance_id, utterance_symbols in utterances.items():
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

  __remove_utterances_and_selection_with_logging(remove, utterances, selection)


def merge(data: List[Tuple[Selection, ReadingPassages, Representations]]) -> Tuple[Selection, ReadingPassages, Representations]:
  assert len(data) > 0

  first_entry = data[0]

  first_selection, first_reading_passages, first_representations = first_entry

  merged_selection = Selection()

  merged_reading_passages = ReadingPassages()
  merged_reading_passages.language = first_reading_passages.language
  merged_reading_passages.symbol_format = first_reading_passages.symbol_format

  merged_representations = Representations()
  merged_representations.language = first_representations.language
  merged_representations.symbol_format = first_representations.symbol_format

  for data_to_merge in data:
    current_selection, current_reading_passages, current_representations = data_to_merge

    assert len(current_reading_passages) == len(current_representations)
    assert merged_reading_passages.language == current_reading_passages.language
    assert merged_reading_passages.symbol_format == current_reading_passages.symbol_format
    assert merged_representations.language == merged_representations.language
    assert merged_representations.symbol_format == merged_representations.symbol_format

    for utterance_id in current_reading_passages.keys():
      next_merged_utterance_id = len(merged_reading_passages)
      merged_reading_passages[next_merged_utterance_id] = current_reading_passages[utterance_id]
      merged_representations[next_merged_utterance_id] = current_representations[utterance_id]
      if utterance_id in current_selection:
        merged_selection.add(next_merged_utterance_id)

  return merged_selection, merged_reading_passages, merged_representations


def select_kld_ngrams_iterations(utterances: Utterances, selection: Selection, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]):
  logger = getLogger(__name__)
  currently_not_selected_utterances = OrderedDict({
    utterance_id: symbols for utterance_id, symbols in utterances.items()
      if utterance_id not in selection
  })

  new_selected = greedy_kld_uniform_ngrams_iterations(
    data=currently_not_selected_utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    iterations=iterations,
  )

  selection |= new_selected

  logger.info(f"Added {len(new_selected)} utterances to selection.")


# def filter_after_duration(corpus: Dict[int, float], min_duration_incl: float, max_duration_excl: float) -> Set[int]:
#   assert min_duration_incl >= 0
#   assert max_duration_excl >= 0

#   filtered_utterance_indicies = set()

#   for utterance_id, utterance_duration in corpus.items():
#     if min_duration_incl <= utterance_duration < max_duration_excl:
#       filtered_utterance_indicies.add(utterance_id)

#   return filtered_utterance_indicies


def select_greedy_ngrams_epochs(utterances: Utterances, selection: Selection, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]]) -> None:
  logger = getLogger(__name__)

  deselected_utterances = get_deselected_utterances(utterances, selection)

  new_selected = greedy_ngrams_epochs(
    data=deselected_utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )

  selection |= new_selected

  logger.info(f"Added {len(new_selected)} utterances to selection.")


def select_from_tex(utterances: Utterances, selection: Selection, tex: str) -> None:
  logger = getLogger(__name__)
  ids_in_tex = detect_ids_from_tex(tex)

  new_ids = ids_in_tex - selection
  if len(new_ids) > 0:
    logger.error("Added new entries:")
    logger.info(new_ids)
    raise Exception()

  #final_ids = selection.intersection(ids_in_tex)
  final_ids = OrderedSet()
  for current_id in selection:
    if current_id in ids_in_tex:
      final_ids.add(current_id)

  removed_count = len(selection - ids_in_tex)

  if removed_count == 0:
    logger.info("Nothing to do.")
  else:
    old_len = len(selection)
    remove_ids = selection - ids_in_tex
    selection -= remove_ids
    if old_len == 0:
      logger.info(
        f"Removed {len(remove_ids)} utterances from selection (100%).")
    else:
      # ids = ','.join(list(map(str, list(sorted(remove_ids)))))
      logger.info(
        f"Removed {len(remove_ids)} utterances from selection ({len(remove_ids)/old_len*100:.2}%).")
    for removed_id in sorted(remove_ids):
      logger.info(f"- {removed_id}: {''.join(utterances[removed_id])}")


def get_utterance_durations_based_on_symbols(utterances: Utterances, reading_speed_symbols_per_s: float) -> Dict[UtteranceId, float]:
  durations = {
    utterance_id: len(symbols) / reading_speed_symbols_per_s
    for utterance_id, symbols in utterances.items()
  }
  return durations


def get_deselected_utterances(utterances: Utterances, selection: Selection) -> Utterances:
  deselected_utterances = Utterances({
    utterance_id: symbols for utterance_id, symbols in utterances.items()
      if utterance_id not in selection
  })
  return deselected_utterances


def get_selected_utterances(utterances: Utterances, selection: Selection) -> Utterances:
  selected_utterances = Utterances({
    utterance_id: symbols for utterance_id, symbols in utterances.items()
      if utterance_id in selection
  })
  return selected_utterances


def select_kld_ngrams_duration(utterances: Utterances, selection: Selection, n_gram: int, minutes: float, utterance_durations_s: Dict[UtteranceId, float], ignore_symbols: Set[Symbol], boundary: DurationBoundary) -> None:
  logger = getLogger(__name__)

  selected_utterances = get_selected_utterances(utterances, selection)
  deselected_utterances = get_deselected_utterances(utterances, selection)
  logger.info(f"Already selected: {len(selected_utterances)}.")

  newly_selected = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=deselected_utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=minutes * 60,
    durations_s=utterance_durations_s,
    preselection=selected_utterances,
    duration_boundary=boundary,
    mp=True,
  )

  selection |= newly_selected
  newly_selected_duration_s = sum(
    duration_s
    for utterance_id, duration_s in utterance_durations_s.items()
    if utterance_id in newly_selected
  )

  logger.info(
    f"Added {len(newly_selected)} utterances to selection ({newly_selected_duration_s/60:.2f}min).")


def select_greedy_ngrams_duration(utterances: Utterances, selection: Selection, n_gram: int, minutes: float, utterance_durations_s: Dict[UtteranceId, float], ignore_symbols: Optional[Set[Symbol]], mode: SelectionMode) -> None:
  logger = getLogger(__name__)

  selected_utterances = get_selected_utterances(utterances, selection)
  deselected_utterances = get_deselected_utterances(utterances, selection)
  logger.info(f"Already selected: {len(selected_utterances)}.")

  newly_selected = greedy_ngrams_durations_advanced(
    data=deselected_utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    target_duration=minutes * 60,
    durations=utterance_durations_s,
    mode=mode,
  )

  selection |= newly_selected
  newly_selected_duration_s = sum(
    duration_s
    for utterance_id, duration_s in utterance_durations_s.items()
    if utterance_id in newly_selected
  )

  logger.info(
    f"Added {len(newly_selected)} utterances to selection ({newly_selected_duration_s/60:.2f}min).")
