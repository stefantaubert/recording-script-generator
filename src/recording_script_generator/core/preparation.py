from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger
from typing import List, Optional
from typing import OrderedDict as OrderedDictType

from text_utils import Language
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import (EngToIpaMode, text_normalize, text_to_ipa,
                             text_to_symbols)
from tqdm import tqdm


class PreparationTarget(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1
  BOTH = 2


@dataclass
class PreparationData:
  reading_passages_lang: Language
  reading_passages: OrderedDictType[int, List[str]]
  representations_lang: Language
  representations: OrderedDictType[int, List[str]]


def add_corpus_from_text(utterances: List[str], lang: Language, ipa_settings: Optional[IPAExtractionSettings]) -> PreparationData:
  logger = getLogger(__name__)
  reading_passages: OrderedDictType[int, List[str]] = OrderedDict()
  for i, utt in enumerate(utterances):
    symbols = text_to_symbols(
      text=utt,
      lang=lang,
      ipa_settings=ipa_settings,
      logger=logger,
    )
    reading_passages[i] = symbols

  representation: OrderedDictType[int, List[str]] = reading_passages.copy()
  res = PreparationData(
    reading_passages_lang=lang,
    reading_passages=reading_passages,
    representations=representation,
    representations_lang=lang,
  )
  return res


def _normalize_target(target: OrderedDictType[int, List[str]], lang: Language, ipa_settings: Optional[IPAExtractionSettings]):
  logger = getLogger(__name__)
  for i, symbols in target.items():
    text = ''.join(symbols)
    normalized_text = text_normalize(text, lang, logger)
    symbols = text_to_symbols(normalized_text, lang, ipa_settings, logger)
    target[i] = symbols


def normalize(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings]):
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    _normalize_target(data.reading_passages, data.reading_passages_lang, ipa_settings)
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    _normalize_target(data.representations, data.representations_lang, ipa_settings)
  return data


def _convert_to_ipa_target(target: OrderedDictType[int, List[str]], lang: Language, ipa_settings: Optional[IPAExtractionSettings], mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str], use_cache: Optional[bool]):
  logger = getLogger(__name__)
  if lang == Language.IPA:
    logger.info("Text is already IPA.")
    return
  for i, symbols in tqdm(target.items()):
    text = ''.join(symbols)
    ipa_text = text_to_ipa(
      text=text,
      lang=lang,
      mode=mode,
      replace_unknown_with=replace_unknown_with,
      use_cache=use_cache,
      logger=logger
    )
    symbols = text_to_symbols(ipa_text, Language.IPA, ipa_settings, logger)
    target[i] = symbols


def convert_to_ipa(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings], mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str], use_cache: Optional[bool]):
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    _convert_to_ipa_target(data.reading_passages, data.reading_passages_lang,
                           ipa_settings, mode, replace_unknown_with, use_cache)
    data.reading_passages_lang = Language.IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    _convert_to_ipa_target(data.representations, data.representations_lang,
                           ipa_settings, mode, replace_unknown_with, use_cache)
    data.representations_lang = Language.IPA
  return data
