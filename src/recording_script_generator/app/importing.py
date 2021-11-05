from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

from general_utils.main import get_all_files_in_all_subfolders
from recording_script_generator.app.io import *
from recording_script_generator.core.importing import (
    add_corpus_from_text_files, add_corpus_from_texts, merge)
from text_utils import Language
from text_utils.symbol_format import SymbolFormat


def app_add_corpus_from_text_files(base_dir: Path, corpus_name: str, step_name: str, text_dir: Path, lang: Language, text_format: SymbolFormat, limit: Optional[int], chunksize_files: int, chunksize_utterances: int, n_jobs: int, overwrite: bool = False) -> None:
  logger = getLogger(__name__)

  if not text_dir.is_dir():
    logger.error("Text folder does not exist.")
    return

  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  logger.info("Detecting all .txt files...")
  all_files = get_all_files_in_all_subfolders(text_dir)
  all_txt_files = {file for file in all_files if file.suffix.lower() == ".txt"}
  logger.info(f"Detected {len(all_txt_files)} .txt files.")

  selection, reading_passages, representations = add_corpus_from_text_files(
    files=all_txt_files,
    lang=lang,
    text_format=text_format,
    limit=limit,
    n_jobs=n_jobs,
    chunksize_files=chunksize_files,
    chunksize_utterances=chunksize_utterances,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Done.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_data(step_dir, selection, reading_passages, representations)


def app_add_corpus_from_text_file(base_dir: Path, corpus_name: str, step_name: str, text_path: Path, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  if not text_path.exists():
    logger = getLogger(__name__)
    logger.error("File does not exist.")
    return

  text = text_path.read_text()
  app_add_corpus_from_text(base_dir, corpus_name, step_name, text, lang, text_format, overwrite)


def app_add_corpus_from_text(base_dir: Path, corpus_name: str, step_name: str, text: str, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  selection, reading_passages, representations = add_corpus_from_texts(
    texts=[text],
    lang=lang,
    text_format=text_format,
    n_jobs=1,
    chunksize=1,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_data(step_dir, selection, reading_passages, representations)


def app_merge(base_dir: Path, corpora_step_names: List[Tuple[str, str]], out_corpus_name: str, out_step_name: str, overwrite: bool = False):
  logger = getLogger(__name__)
  logger.info(f"Merging multiple corpora...")
  assert corpora_step_names is not None

  out_corpus_dir = get_corpus_dir(base_dir, out_corpus_name)
  if out_corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  data_to_merge: List[Tuple[Selection, ReadingPassages, Representations]] = []
  for corpus_name, in_step_name in corpora_step_names:
    in_corpus_dir = get_corpus_dir(base_dir, corpus_name)
    in_step_dir = get_step_dir(in_corpus_dir, in_step_name)

    if not in_step_dir.exists():
      logger.error("In step dir does not exist!")
      return

    loaded_data = load_data(in_step_dir)
    data_to_merge.append(loaded_data)

  merged_selection, merged_reading_passages, merged_representations = merge(data_to_merge)

  if out_corpus_dir.exists():
    assert overwrite
    rmtree(out_corpus_dir)
    logger.info("Removed existing corpus.")

  out_corpus_dir.mkdir(parents=True, exist_ok=False)
  out_step_dir = get_step_dir(out_corpus_dir, out_step_name)
  save_data(out_step_dir, merged_selection, merged_reading_passages, merged_representations)
