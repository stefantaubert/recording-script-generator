from collections import OrderedDict
from copy import deepcopy
from logging import getLogger
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

from recording_script_generator.core.types import (Paths, ReadingPassages,
                                                   ReadingPassagesPaths,
                                                   Representations, Selection,
                                                   Utterance, UtteranceId,
                                                   Utterances)
from text_utils import SymbolFormat, text_to_sentences
from text_utils.language import Language
from text_utils.text import symbols_to_sentences
from tqdm import tqdm


def main_inplace(selection: Selection, reading_passages: ReadingPassages, reading_passages_paths: ReadingPassagesPaths, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int]) -> Representations:
  logger = getLogger(__name__)
  sentences = split_sentences(reading_passages, n_jobs, maxtasksperchild, chunksize, None)

  new_reading_passages = ReadingPassages()
  new_reading_passages_paths = Paths()
  new_selection = Selection()

  logger.info("Processing sentences...")
  for utterance_id in tqdm(reading_passages.keys()):
    utterance_was_selected = utterance_id in selection
    utterance_path = reading_passages_paths[utterance_id]
    for utterance_sentence in sentences[utterance_id]:
      new_utterance_id = len(new_reading_passages)
      new_reading_passages[new_utterance_id] = utterance_sentence
      new_reading_passages_paths[new_utterance_id] = utterance_path
      if utterance_was_selected:
        new_selection.add(new_utterance_id)

  logger.info("Updating reading passages...")
  reading_passages.clear()
  reading_passages.update(new_reading_passages)
  
  logger.info("Updating reading passage paths...")
  reading_passages_paths.clear()
  reading_passages_paths.update(new_reading_passages_paths)

  logger.info("Updating selection...")
  selection.clear()
  for selected_utterance_id in new_selection:
    selection.add(selected_utterance_id)

  logger.info("Updating representations...")
  new_representations = deepcopy(reading_passages)
  logger.info("Done.")

  return new_representations


def split_sentences(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Dict[UtteranceId, Set[Utterance]]:
  logger = getLogger(__name__)
  logger.info("Extracting sentences...")
  logger.info(f"Assigning {chunksize} files to {n_jobs} processor core.")

  with Pool(
    processes=n_jobs,
    initializer=init_pool,
    initargs=(utterances,),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    sentences_from_utterances: Dict[UtteranceId, Set[Utterance]] = OrderedDict(tqdm(
      pool.imap_unordered(extract_sentences, utterances.keys(), chunksize=chunksize),
      total=len(utterances),
    ))

  logger.info("Done.")
  return sentences_from_utterances


process_utterances: Utterances = None


def init_pool(utterances: Utterances) -> None:
  global process_utterances
  process_utterances = utterances


def extract_sentences(utterance_id: UtteranceId) -> Tuple[UtteranceId, Set[Utterance]]:
  # pylint: disable=global-variable-not-assigned
  global process_utterances
  utterance = process_utterances[utterance_id]
  potential_sentences = utterance_to_sentences(
    utterance, process_utterances.language, process_utterances.symbol_format)
  sentences: Set[str] = set()

  for potential_sentence in potential_sentences:
    if not is_sentence(potential_sentence, process_utterances.language, process_utterances.symbol_format):
      continue

    sentences.add(potential_sentence)
  return utterance_id, potential_sentences


def utterance_to_sentences(utterance: Utterance, lang: Language, symbol_format: SymbolFormat) -> List[Utterance]:
  if isinstance(utterance, tuple):
    return symbols_to_sentences(utterance, symbol_format, lang)
  if isinstance(utterance, str):
    return text_to_sentences(utterance, symbol_format, lang)
  assert False


def ends_with_punctuation(utterance: Utterance) -> bool:
  if len(utterance) == 0:
    return False

  last_letter = utterance[-1]

  # TODO: multilanguage support
  contains_sentence_ending = last_letter in {".", "!", "?"}
  if not contains_sentence_ending:
    return False

  return True


def starts_with_big_letter(utterance: Utterance) -> bool:
  if len(utterance) == 0:
    return False

  first_letter = utterance[0]
  first_letter_is_upper = first_letter.isupper()
  return first_letter_is_upper


def is_sentence(utterance: Utterance, language: Language, symbol_format: SymbolFormat) -> bool:
  if symbol_format == SymbolFormat.GRAPHEMES:
    if language == Language.ENG or language == Language.GER:
      return starts_with_big_letter(utterance) and ends_with_punctuation(utterance)
    else:
      raise Exception("Not supported!")
  if symbol_format.is_IPA:
    return ends_with_punctuation(utterance)
  if symbol_format.is_ARPA:
    raise Exception("Not supported!")
  assert False
