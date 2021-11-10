from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   Utterance, UtteranceId,
                                                   Utterances)
from text_utils import Language, SymbolFormat
from text_utils.text import text_to_sentences
from tqdm import tqdm


def read_textfile(path: Path) -> str:
  content = path.read_text()
  content = content.replace("\n", " ")
  return content


def add_corpus_from_text_files(files: Set[Path], lang: Language, text_format: SymbolFormat, limit: Optional[int], chunksize: int, n_jobs: int, maxtasksperchild: Optional[int]) -> Tuple[Selection, ReadingPassages, Representations]:
  logger = getLogger(__name__)
  logger.info("Reading text files...")
  if limit is not None:
    files = set(list(files)[:limit])
  with ThreadPoolExecutor(max_workers=n_jobs) as ex:
    res = list(tqdm(ex.map(read_textfile, files, chunksize=chunksize), total=len(files)))
  logger.info("Done.")

  return add_corpus_from_texts(
    texts=res,
    language=lang,
    text_format=text_format,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
  )


def get_sentences_from_text(text: str, lang: Language, text_format: SymbolFormat) -> Set[str]:
  utterances = file_to_utterances(text, lang, text_format)
  sentences: Set[str] = set()

  for utterance in utterances:
    if not is_sentence(utterance, lang, text_format):
      continue

    # symbols = text_to_symbols(
    #   text=utterance,
    #   lang=lang,
    #   text_format=text_format,
    # )

    sentences.add(utterance)
  return sentences


process_texts: List[str] = None


def init_pool(texts: List[str]) -> None:
  global process_texts
  process_texts = texts


def get_sentences_from_text_v2(text_nr: int, language: Language, text_format: SymbolFormat) -> Set[str]:
  # pylint: disable=global-variable-not-assigned
  global process_texts
  text = process_texts[text_nr]
  utterances = file_to_utterances(text, language, text_format)
  sentences: Set[str] = set()

  for utterance in utterances:
    if not is_sentence(utterance, language, text_format):
      continue

    # symbols = text_to_symbols(
    #   text=utterance,
    #   lang=lang,
    #   text_format=text_format,
    # )

    sentences.add(utterance)
  return sentences


def add_corpus_from_texts(texts: List[str], language: Language, text_format: SymbolFormat, chunksize: int, n_jobs: int, maxtasksperchild: Optional[int]) -> Tuple[Selection, ReadingPassages, Representations]:
  logger = getLogger(__name__)
  method_proxy = partial(get_sentences_from_text, lang=language, text_format=text_format)
  logger.info("Detecting valid sentences...")
  #tqdm_steps = 4
  #chunksize = max(round(len(texts) / n_jobs / tqdm_steps), 1)
  logger.info(f"Assigning {chunksize} files to {n_jobs} processor core.")

  method_proxy = partial(
    get_sentences_from_text_v2,
    language=language,
    text_format=text_format,
  )

  with Pool(
    processes=n_jobs,
    initializer=init_pool,
    initargs=(texts,),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    sentences_from_files: List[Set[str]] = list(tqdm(
      pool.imap_unordered(method_proxy, range(len(texts)), chunksize=chunksize),
      total=len(texts),
    ))

  # # todo optimize that texts are not passed as argument
  # with ProcessPoolExecutor(max_workers=n_jobs) as ex:
  #   sentences_from_files: List[Set[str]] = list(
  #     tqdm(ex.map(method_proxy, texts, chunksize=chunksize), total=len(texts)))
  logger.info("Done.")
  logger.info("Extracting sentences...")
  all_sentences: Set[str] = {
    text_sentence
    for text_sentences in tqdm(sentences_from_files)
    for text_sentence in text_sentences
  }

  reading_passages = ReadingPassages({
    utterance_id: sentence for utterance_id, sentence in enumerate(tqdm(all_sentences))
  })
  reading_passages.symbol_format = text_format
  reading_passages.language = language

  # selected_percent = len(reading_passages) / total_utterance_count
  # logger.info(
  #   f"{selected_percent*100:.2f}% ({len(reading_passages)}) of all {total_utterance_count} utterances were sentences and thus selected.")
  logger.info("Cloning as representations...")
  representations = Representations(reading_passages)
  representations.symbol_format = text_format
  representations.language = language
  logger.info(f"Done. Detected {len(reading_passages)} sentences.")

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

def file_to_utterances(content: str, lang: Language, text_format: SymbolFormat) -> List[str]:
  content_lines = content.split("\n")
  res = []
  for line in content_lines:
    sentences = text_to_sentences(
      text=line.strip(),
      lang=lang,
      text_format=text_format,
    )
    res.extend(sentences)
  return res


def ends_with_punctuation(sentence: str) -> bool:
  if len(sentence) == 0:
    return False

  last_letter = sentence[-1]

  contains_sentence_ending = last_letter in {".", "!", "?"}
  if not contains_sentence_ending:
    return False

  return True


def starts_with_big_letter(sentence: str) -> bool:
  if len(sentence) == 0:
    return False

  first_letter = sentence[0]
  res = first_letter.isupper()
  return res


def is_sentence(sentence: str, lang: Language, sentence_format: SymbolFormat) -> bool:
  if sentence_format == SymbolFormat.GRAPHEMES:
    if lang == Language.ENG or lang == Language.GER:
      return starts_with_big_letter(sentence) and ends_with_punctuation(sentence)
    else:
      raise Exception("Not supported!")
  elif sentence_format.is_IPA:
    return ends_with_punctuation(sentence)
  else:
    assert False
