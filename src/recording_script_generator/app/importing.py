from collections import OrderedDict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

from general_utils.main import get_all_files_in_all_subfolders
from ordered_set import OrderedSet
from recording_script_generator.app.helper import (
    get_text_files_tqdm, raise_error_if_directory_exists_and_not_overwrite,
    raise_error_if_directory_not_exists)
from recording_script_generator.app.io import *
from recording_script_generator.core.importing import (
    add_corpus_from_text_files, add_corpus_from_texts, create_from_both_files,
    merge)
from text_utils import Language
from text_utils.string_format import StringFormat
from text_utils.symbol_format import SymbolFormat


def app_add_files(working_directory: Path, directory: Path, language: Language = Language.ENG, symbol_format: SymbolFormat = SymbolFormat.GRAPHEMES, string_format: StringFormat = StringFormat.TEXT, custom_representations_string_format: Optional[StringFormat] = None, custom_representations_directory: Optional[Path] = None, custom_representations_symbol_format: Optional[SymbolFormat] = None, limit: Optional[int] = None, overwrite: bool = False) -> None:
  if raise_error_if_directory_not_exists(directory):
    return

  representations_directory = directory
  if custom_representations_directory is not None:
    if raise_error_if_directory_not_exists(custom_representations_directory):
      return
    representations_directory = custom_representations_directory

  if raise_error_if_directory_exists_and_not_overwrite(working_directory, overwrite):
    return

  representations_symbol_format = symbol_format
  if custom_representations_symbol_format is not None:
    representations_symbol_format = custom_representations_symbol_format

  representations_string_format = string_format
  if custom_representations_string_format is not None:
    representations_string_format = custom_representations_string_format

  logger = getLogger(__name__)

  logger.info("Detecting all reading passage files...")
  all_reading_passage_files = get_text_files_tqdm(directory)
  logger.info(f"Detected {len(all_reading_passage_files)} reading passage files.")

  if representations_directory == directory:
    all_representation_files = deepcopy(all_reading_passage_files)
  else:
    logger.info("Detecting all representation files...")
    all_representation_files = get_text_files_tqdm(representations_directory)
  logger.info(f"Detected {len(all_representation_files)} representation files.")

  common_files = OrderedSet(all_reading_passage_files.keys()
                            ).intersection(all_representation_files.keys())
  logger.info(f"{len(common_files)} matching files.")

  absolute_paths = [
    (directory / all_reading_passage_files[file_stem], representations_directory / all_representation_files[file_stem]) for file_stem in common_files
  ]

  created_data = create_from_both_files(
    absolute_paths,
    language=language,
    limit=limit,
    symbol_formats=(symbol_format, representations_symbol_format),
    string_formats=(string_format, representations_string_format),
  )

  selection, reading_passages, representations, absolute_paths, reading_passages_paths = created_data

  if working_directory.exists():
    assert overwrite
    logger.info("Removing existing corpus ...")
    rmtree(working_directory)
    logger.info("Done.")

  save_data(working_directory, selection, reading_passages,
            representations, absolute_paths, reading_passages_paths)


def app_add_corpus_from_text_files(base_dir: Path, corpus_name: str, step_name: str, text_dir: Path, lang: Language, text_format: SymbolFormat, limit: Optional[int], chunksize: int, n_jobs: int, maxtasksperchild: Optional[int], overwrite: bool = False) -> None:
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

  selection, reading_passages, representations, utterance_paths = add_corpus_from_text_files(
    files=all_txt_files,
    lang=lang,
    text_format=text_format,
    limit=limit,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Done.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_data(step_dir, selection, reading_passages, representations, utterance_paths)


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

  selection, reading_passages, representations, utterance_paths = add_corpus_from_texts(
    texts=[(None, text)],
    language=lang,
    text_format=text_format,
    n_jobs=1,
    chunksize=1,
    maxtasksperchild=None,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_data(step_dir, selection, reading_passages, representations, utterance_paths)


def app_merge(base_dir: Path, corpora_step_names: List[Tuple[str, str]], out_corpus_name: str, out_step_name: str, overwrite: bool = False):
  logger = getLogger(__name__)
  logger.info(f"Merging multiple corpora...")
  assert corpora_step_names is not None

  out_corpus_dir = get_corpus_dir(base_dir, out_corpus_name)
  if out_corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  data_to_merge: List[Tuple[Selection, ReadingPassages, Representations, ReadingPassagesPaths]] = []
  for corpus_name, in_step_name in corpora_step_names:
    in_corpus_dir = get_corpus_dir(base_dir, corpus_name)
    in_step_dir = get_step_dir(in_corpus_dir, in_step_name)

    if not in_step_dir.exists():
      logger.error("In step dir does not exist!")
      return

    loaded_data = load_data(in_step_dir)
    data_to_merge.append(loaded_data)

  merged_selection, merged_reading_passages, merged_representations, merged_utterance_paths = merge(
    data_to_merge)

  if out_corpus_dir.exists():
    assert overwrite
    rmtree(out_corpus_dir)
    logger.info("Removed existing corpus.")

  out_corpus_dir.mkdir(parents=True, exist_ok=False)
  out_step_dir = get_step_dir(out_corpus_dir, out_step_name)
  save_data(out_step_dir, merged_selection, merged_reading_passages,
            merged_representations, merged_utterance_paths)
