from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from recording_script_generator.core.text_extraction import (
    file_to_utterances, is_sentence)
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   UtteranceId, Utterances)
from text_utils import Language, SymbolFormat, Symbols, text_to_symbols
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


# def filter_after_duration(corpus: Dict[int, float], min_duration_incl: float, max_duration_excl: float) -> Set[int]:
#   assert min_duration_incl >= 0
#   assert max_duration_excl >= 0

#   filtered_utterance_indicies = set()

#   for utterance_id, utterance_duration in corpus.items():
#     if min_duration_incl <= utterance_duration < max_duration_excl:
#       filtered_utterance_indicies.add(utterance_id)

#   return filtered_utterance_indicies


def get_utterance_durations_based_on_symbols(utterances: Utterances, reading_speed_symbols_per_s: float) -> Dict[UtteranceId, float]:
  durations = {
    utterance_id: len(symbols) / reading_speed_symbols_per_s
    for utterance_id, symbols in utterances.items()
  }
  return durations
