from argparse import ArgumentParser
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Generator, List, Optional, Tuple

from ordered_set import OrderedSet
from recording_script_generator.app.helper import (
    add_overwrite_argument, get_text_files_tqdm,
    raise_error_if_directory_exists_and_not_overwrite,
    raise_error_if_directory_not_exists)
from recording_script_generator.app.io import *
from recording_script_generator.core.importing import (create_from_both_files2,
                                                       merge)
from text_utils import Language
from text_utils.string_format import StringFormat
from text_utils.symbol_format import SymbolFormat


def init_app_add_files_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('directory', type=Path,
                      help="directory containing the text files which should be added")
  parser.add_argument('--language', choices=Language,
                      type=Language.__getitem__, default=Language.ENG)
  parser.add_argument('--symbol-format', choices=SymbolFormat,
                      type=SymbolFormat.__getitem__, default=SymbolFormat.GRAPHEMES)
  parser.add_argument('--string-format', choices=StringFormat,
                      type=StringFormat.__getitem__, default=StringFormat.TEXT)
  parser.add_argument('--limit', type=int, default=None)
  parser.add_argument('--custom-representations-directory', type=Path, default=None)
  parser.add_argument('--custom-representations-symbol-format', choices=SymbolFormat,
                      type=SymbolFormat.__getitem__, default=None)
  parser.add_argument('--custom-representations-string-format', choices=StringFormat,
                      type=StringFormat.__getitem__, default=None)
  parser.add_argument('--encoding', type=str, default="utf-8")
  add_overwrite_argument(parser)
  return app_add_files


def app_add_files(working_directory: Path, directory: Path, encoding: str, language: Language = Language.ENG, symbol_format: SymbolFormat = SymbolFormat.GRAPHEMES, string_format: StringFormat = StringFormat.TEXT, custom_representations_string_format: Optional[StringFormat] = None, custom_representations_directory: Optional[Path] = None, custom_representations_symbol_format: Optional[SymbolFormat] = None, limit: Optional[int] = None, overwrite: bool = False) -> None:
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

  created_data = create_from_both_files2(
    absolute_paths,
    language=language,
    limit=limit,
    symbol_formats=(symbol_format, representations_symbol_format),
    string_formats=(string_format, representations_string_format),
    encoding=encoding,
  )

  selection, reading_passages, representations, absolute_paths, reading_passages_paths = created_data

  if working_directory.exists():
    assert overwrite
    logger.info("Removing existing corpus ...")
    rmtree(working_directory)
    logger.info("Done.")

  save_data(working_directory, selection, reading_passages,
            representations, absolute_paths, reading_passages_paths)


# def init_merge_parser(parser: ArgumentParser):
#   parser.add_argument('--corpora_step_names', type=str, required=True)
#   parser.add_argument('--out_corpus_name', type=str, required=True)
#   parser.add_argument('--out_step_name', type=str, required=True)
#   parser.add_argument('--overwrite', action='store_true')
#   return _merge_cli


# def _merge_cli(**args):
#   args["corpora_step_names"] = parse_tuple_list(args["corpora_step_names"])
#   app_merge(**args)

# def app_merge(base_dir: Path, corpora_step_names: List[Tuple[str, str]], out_corpus_name: str, out_step_name: str, overwrite: bool = False):
#   logger = getLogger(__name__)
#   logger.info(f"Merging multiple corpora...")
#   assert corpora_step_names is not None

#   out_corpus_dir = get_corpus_dir(base_dir, out_corpus_name)
#   if out_corpus_dir.exists() and not overwrite:
#     logger.info("Corpus already exists.")
#     return

#   data_to_merge: List[Tuple[Selection, ReadingPassages, Representations, ReadingPassagesPaths]] = []
#   for corpus_name, in_step_name in corpora_step_names:
#     in_corpus_dir = get_corpus_dir(base_dir, corpus_name)
#     in_step_dir = get_step_dir(in_corpus_dir, in_step_name)

#     if not in_step_dir.exists():
#       logger.error("In step dir does not exist!")
#       return

#     loaded_data = load_data(in_step_dir)
#     data_to_merge.append(loaded_data)

#   merged_selection, merged_reading_passages, merged_representations, merged_utterance_paths = merge(
#     data_to_merge)

#   if out_corpus_dir.exists():
#     assert overwrite
#     rmtree(out_corpus_dir)
#     logger.info("Removed existing corpus.")

#   out_corpus_dir.mkdir(parents=True, exist_ok=False)
#   out_step_dir = get_step_dir(out_corpus_dir, out_step_name)
#   save_data(out_step_dir, merged_selection, merged_reading_passages,
#             merged_representations, merged_utterance_paths)
