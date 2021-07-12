from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, Union

import enchant
from numpy import select
from ordered_set import OrderedSet
from pandas import DataFrame
from readability import Readability
from recording_script_generator.core.text_extraction import (
    contains_proper_names, contains_undesired_text, file_to_text,
    get_minimum_frequency, get_non_dict_words_amount, is_sentence,
    remove_undesired_en_sentences, strip_punctuation_words,
    words_contain_acronyms)
from text_selection import greedy_ngrams_epochs
from text_utils import Language
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import (EngToIpaMode, text_normalize, text_to_ipa,
                             text_to_symbols)
from tqdm import tqdm

SentenceId = int
Symbols = List[str]
ReadingPassage = Symbols
Representation = Symbols
ReadingPassages = Dict[int, ReadingPassage]
Representations = Dict[int, Representation]


class PreparationTarget(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1
  BOTH = 2


class Mode(IntEnum):
  SELECT = 0
  DESELECT = 1


@dataclass()
class PreparationData:
  reading_passages_lang: Language
  reading_passages: ReadingPassages
  representations_lang: Language
  representations: Representations
  selected: OrderedSet[SentenceId]


def add_corpus_from_text(text: str, lang: Language, ipa_settings: Optional[IPAExtractionSettings]) -> PreparationData:
  logger = getLogger(__name__)
  reading_passages: Dict[SentenceId, ReadingPassage] = {}
  utterances = file_to_text(text, lang=lang)

  for utterance_id, utterance in enumerate(utterances):
    if not is_sentence(utterance, lang):
      continue

    symbols = text_to_symbols(
      text=utterance,
      lang=lang,
      ipa_settings=ipa_settings,
      logger=logger,
    )

    reading_passages[utterance_id] = symbols

  selected_percent = len(reading_passages) / len(utterances)
  logger.info(
    f"{selected_percent*100:.2f}% ({len(reading_passages)}) of all {len(utterances)} utterances were sentences and thus selected.")

  result = PreparationData(
    reading_passages_lang=lang,
    reading_passages=reading_passages,
    representations=reading_passages.copy(),
    representations_lang=lang,
    selected=OrderedSet(),
  )

  return result


def normalize(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings]) -> None:
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Normalizing reading passages...")
    target = data.reading_passages
    lang = data.reading_passages_lang
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Normalizing representations...")
    target = data.representations
    lang = data.representations_lang

  for utterance_id, symbols in tqdm(target.items()):
    text = ''.join(symbols)
    normalized_text = text_normalize(text, lang, logger)
    symbols = text_to_symbols(normalized_text, lang, ipa_settings, logger)
    target[utterance_id] = symbols
  logger.info("Done.")


def convert_to_ipa(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings], mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str], consider_ipa_annotations: bool, use_cache: Optional[bool]) -> None:
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Converting reading passages to IPA...")
    target = data.reading_passages
    lang = data.reading_passages_lang
    data.reading_passages_lang = Language.IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Converting representations to IPA...")
    target = data.representations
    lang = data.representations_lang
    data.representations_lang = Language.IPA

  if lang == Language.IPA:
    logger.info("Text is already IPA.")
    return

  for utterance_id, symbols in tqdm(target.items()):
    text = ''.join(symbols)
    ipa_text = text_to_ipa(
      text=text,
      lang=lang,
      mode=mode,
      replace_unknown_with=replace_unknown_with,
      use_cache=use_cache,
      consider_ipa_annotations=consider_ipa_annotations,
      logger=logger
    )
    new_symbols = text_to_symbols(ipa_text, Language.IPA, ipa_settings, logger)
    target[utterance_id] = new_symbols


def select_all_utterances(data: PreparationData):
  data.selected |= OrderedSet(data.reading_passages.keys())


def _remove_utterances(utterance_ids: Set[int], data: PreparationData) -> None:
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
  _remove_utterances(utterance_ids, data)
  logger.info(
    f"Removed {len(utterance_ids)} of {old_len} utterances ({len(utterance_ids)/old_len*100:.2f}%) and obtained {len(data.representations)} utterances.")


def remove_utterances_with_undesired_text(data: PreparationData, undesired: Set[str]) -> None:
  remove = OrderedSet()
  for utterance_id, utterance_symbols in data.representations.items():
    utterance = ''.join(utterance_symbols)
    if contains_undesired_text(utterance, undesired=undesired, ignore_case=True):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_duplicate_utterances(data: PreparationData) -> None:
  remove = OrderedSet()
  already_exist: Set[Tuple[str, ...]] = set()
  for utterance_id, utterance_symbols in data.representations.items():
    utterance_symbols_tuple = tuple(utterance_symbols)
    if utterance_symbols_tuple in already_exist:
      remove.add(utterance_id)
    else:
      already_exist.add(utterance_symbols_tuple)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_proper_names_and_acronyms(data: PreparationData) -> None:
  remove = OrderedSet()
  for utterance_id, utterance_symbols in data.representations.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    if words_contain_acronyms(words_non_punctuation):
      remove.add(utterance_id)
    elif contains_proper_names(utterance):
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_undesired_sentence_lengths(data: PreparationData, min_word_count: Optional[int], max_word_count: Optional[int]) -> None:
  remove = OrderedSet()
  for utterance_id, utterance_symbols in data.representations.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_count = len(words)

    if min_word_count is not None and words_count < min_word_count:
      remove.add(utterance_id)
    elif max_word_count is not None and words_count > max_word_count:
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_unknown_words(data: PreparationData, max_unknown_word_count: int) -> None:
  if data.representations_lang != Language.ENG:
    logger = getLogger(__name__)
    logger.error("Language needs to be English!")
    return

  lexicon = enchant.Dict("en_US")
  remove = OrderedSet()
  for utterance_id, utterance_symbols in data.representations.items():
    utterance = ''.join(utterance_symbols)
    words = utterance.split(" ")
    words_non_punctuation = strip_punctuation_words(words)

    non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, lexicon)
    if non_dict_words_amount > max_unknown_word_count:
      remove.add(utterance_id)

  _remove_utterances_with_logging(remove, data)


def remove_utterances_with_too_seldom_words(data: PreparationData, min_occurrence_count: int) -> None:
  remove = OrderedSet()
  stripped_words: Dict[int, List[str]] = {}
  for utterance_id, utterance_symbols in data.representations.items():
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
