from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger
from typing import List, Optional, Tuple

from ordered_set import OrderedSet
from recording_script_generator.core.text_extraction import (
    file_to_text, remove_non_sentences)
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
  reading_passages: List[List[str]]
  representations_lang: Language
  representations: List[List[str]]


def add_corpus_from_text(text: str, lang: Language, ipa_settings: Optional[IPAExtractionSettings]) -> PreparationData:
  logger = getLogger(__name__)
  reading_passages: List[List[str]] = []
  utterances = file_to_text(text, lang=lang)
  unique_entries = OrderedSet(utterances)
  if len(unique_entries) < len(utterances):
    ignored_count = len(utterances) - len(unique_entries)
    logger.info(f"Ignored doubling entries ({ignored_count}/{len(utterances)}).")
  else:
    logger.info("No doubling entries found.")

  if lang == Language.ENG:
    logger.info("Removing undesired English utterances...")
    unique_entries = remove_non_sentences(list(unique_entries))
    logger.info("Done.")
  for utt in unique_entries:
    symbols = text_to_symbols(
      text=utt,
      lang=lang,
      ipa_settings=ipa_settings,
      logger=logger,
    )
    reading_passages.append(symbols)

  representation: List[List[str]] = reading_passages.copy()
  res = PreparationData(
    reading_passages_lang=lang,
    reading_passages=reading_passages,
    representations=representation,
    representations_lang=lang,
  )
  return res


def _normalize_target(target: List[List[str]], lang: Language, ipa_settings: Optional[IPAExtractionSettings]) -> List[List[str]]:
  logger = getLogger(__name__)
  res: List[List[str]] = []
  for symbols in tqdm(target):
    text = ''.join(symbols)
    normalized_text = text_normalize(text, lang, logger)
    symbols = text_to_symbols(normalized_text, lang, ipa_settings, logger)
    res.append(symbols)
  return res


def normalize(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings]):
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Normalizing reading passages...")
    res = _normalize_target(data.reading_passages, data.reading_passages_lang, ipa_settings)
    data.reading_passages = res
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Normalizing representations...")
    res = _normalize_target(data.representations, data.representations_lang, ipa_settings)
    data.representations = res
  return data


def _convert_to_ipa_target(target: List[List[str]], lang: Language, ipa_settings: Optional[IPAExtractionSettings], mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str], consider_ipa_annotations: bool, use_cache: Optional[bool]) -> List[List[str]]:
  logger = getLogger(__name__)
  if lang == Language.IPA:
    logger.info("Text is already IPA.")
    return
  res: List[List[str]] = []
  for symbols in tqdm(target):
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
    symbols = text_to_symbols(ipa_text, Language.IPA, ipa_settings, logger)
    res.append(symbols)
  return res


def convert_to_ipa(data: PreparationData, target: PreparationTarget, ipa_settings: Optional[IPAExtractionSettings], mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str], consider_ipa_annotations: bool, use_cache: Optional[bool]):
  logger = getLogger(__name__)
  if target == PreparationTarget.BOTH or PreparationTarget.READING_PASSAGES:
    logger.info("Converting reading passages to IPA...")
    res = _convert_to_ipa_target(data.reading_passages, data.reading_passages_lang,
                                 ipa_settings, mode, replace_unknown_with, consider_ipa_annotations, use_cache)
    data.reading_passages = res
    data.reading_passages_lang = Language.IPA
  if target == PreparationTarget.BOTH or PreparationTarget.REPRESENTATIONS:
    logger.info("Converting representations to IPA...")
    res = _convert_to_ipa_target(data.representations, data.representations_lang,
                                 ipa_settings, mode, replace_unknown_with, consider_ipa_annotations, use_cache)
    data.representations = res
    data.representations_lang = Language.IPA
  return data
