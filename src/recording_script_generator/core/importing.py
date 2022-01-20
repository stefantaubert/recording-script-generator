from logging import getLogger
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from recording_script_generator.core.types import (Paths, ReadingPassages,
                                                   ReadingPassagesPaths,
                                                   Representations, Selection)
from text_utils import Language, StringFormat, SymbolFormat
from tqdm import tqdm


def read_text(path: Path, encoding: str) -> str:
  assert path.is_file()
  content = path.read_text(encoding)
  return content


def read_lines(path: Path, encoding: str) -> List[str]:
  text = read_text(path, encoding)
  lines = text.split("\n")
  return lines


def str_is_empty_or_whitespace(string: str) -> bool:
  assert string is not None
  result = string.strip() == ""
  return result


def create_from_both_files2(paths: List[Tuple[Path, Path]], symbol_formats: Tuple[SymbolFormat, SymbolFormat], string_formats: Tuple[StringFormat, StringFormat], language: Language, encoding: str, limit: Optional[int]) -> Tuple[Selection, ReadingPassages, Representations, Paths, ReadingPassagesPaths]:
  logger = getLogger(__name__)
  if limit is not None:
    logger.debug(f"Set limit to: {limit}.")
    paths = paths[:limit]

  read_string_format, repr_string_format = string_formats
  read_symbol_format, repr_symbol_format = symbol_formats

  read_paths_are_unique = len(set(x for x, _ in paths)) == len(paths)
  assert read_paths_are_unique

  reading_passages = []
  representations = []
  paths_to_ids = Paths()
  read_paths = []

  logger.info("Reading files...")
  for path_id, (path_read, path_repr) in enumerate(tqdm(paths)):
    lines_read = read_lines(path_read, encoding)
    if path_repr == path_read:
      lines_repr = lines_read
    else:
      lines_repr = read_lines(path_repr, encoding)

    if len(lines_repr) != len(lines_read):
      logger.error(
        f"Lines count does not match for files '{str(path_read)}' ({len(lines_read)}) and '{str(path_repr)}' ({len(lines_repr)})! Skipped.")
      continue

    for line_nr, (line_read, line_repr) in enumerate(zip(lines_read, lines_repr), start=1):
      if str_is_empty_or_whitespace(line_read):
        logger.error(f"Ignoring empty row! File '{str(path_read)}', Line: {line_nr}")
        continue

      if str_is_empty_or_whitespace(line_repr):
        logger.error(f"Ignoring empty line! File '{str(path_repr)}', Line: {line_nr}")
        continue

      if not read_string_format.can_convert_string_to_symbols(line_read):
        logger.error(
          f"Line could not be parsed in the given format! File '{str(path_read)}', Line: {line_nr}")
        continue

      if not repr_string_format.can_convert_string_to_symbols(line_repr):
        logger.error(
          f"Line could not be parsed in the given format! File '{str(path_repr)}', Line: {line_nr}")
        continue

      symbols_read = read_string_format.convert_string_to_symbols(line_read)
      symbols_repr = repr_string_format.convert_string_to_symbols(line_repr)
      symbols_str_read = StringFormat.SYMBOLS.convert_symbols_to_string(symbols_read)
      symbols_str_repr = StringFormat.SYMBOLS.convert_symbols_to_string(symbols_repr)
      if path_id not in paths_to_ids:
        paths_to_ids[path_id] = path_read

      reading_passages.append(symbols_str_read)
      representations.append(symbols_str_repr)
      read_paths.append(path_id)

  res_read = ReadingPassages(enumerate(reading_passages))
  res_read.symbol_format = read_symbol_format
  res_read.language = language
  del reading_passages

  res_repr = ReadingPassages(enumerate(representations))
  res_repr.symbol_format = repr_symbol_format
  res_repr.language = language
  del representations

  utterance_paths = ReadingPassagesPaths(enumerate(read_paths))
  del read_paths

  selection = Selection()

  return selection, res_read, res_repr, paths_to_ids, utterance_paths


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
