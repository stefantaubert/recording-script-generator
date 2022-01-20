from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Optional, Set

from recording_script_generator.app.io import *
from recording_script_generator.core.exporting import (SortingMode,
                                                       df_to_consecutive_txt,
                                                       df_to_paths, df_to_tex,
                                                       df_to_txt,
                                                       generate_textgrid,
                                                       get_df)
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_IGNORE, DEFAULT_SEED,
                                                SEP)
from text_utils.types import Symbol

SELECTED_FILENAME = "selected.csv"
SELECTED_TXT_FILENAME = "selected.txt"
SELECTED_TXT_CONSEC_FILENAME = "selected_consecutive.txt"
SELECTED_TEX_FILENAME = "selected.tex"
SELECTED_PATHS_FILENAME = "selected_paths.txt"

DESELECTED_FILENAME = "deselected.csv"
DESELECTED_TXT_FILENAME = "deselected.txt"
DESELECTED_TXT_CONSEC_FILENAME = "deselected_consecutive.txt"
DESELECTED_TEX_FILENAME = "deselected.tex"
DESELECTED_PATHS_FILENAME = "deselected_paths.txt"

ONE_GRAM_STATS_CSV_FILENAME = "1_gram_stats.csv"
TWO_GRAM_STATS_CSV_FILENAME = "2_gram_stats.csv"
THREE_GRAM_STATS_CSV_FILENAME = "3_gram_stats.csv"


def get_tex_path(step_dir: Path) -> Path:
  return step_dir / SELECTED_TEX_FILENAME


def get_selected_paths_path(step_dir: Path) -> Path:
  return step_dir / SELECTED_PATHS_FILENAME


def init_app_generate_selected_script_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('--sorting_mode', choices=SortingMode,
                      type=SortingMode.__getitem__, default=SortingMode.BY_INDEX)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--parts-count', type=int, default=None)
  parser.add_argument('--take-per-part', type=int, default=None)
  parser.add_argument('--ignore-symbols', type=str, nargs="*", default=[])
  return app_generate_selected_script


def app_generate_selected_script(working_directory: Path, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
  logger = getLogger(__name__)

  if not working_directory.exists():
    logger.info("Corpus does not exist.")
    return

  selection, reading_passages, representations, paths, reading_passages_paths = load_data(
    working_directory)
  logger.info("Saving selected scripts...")

  df = get_df(
    reading_passages=reading_passages,
    representations=representations,
    selection=selection,
    paths=paths,
    read_paths=reading_passages_paths,
    mode=sorting_mode,
    seed=seed,
    ignore_symbols=set(ignore_symbols),
    parts_count=parts_count,
    take_per_part=take_per_part,
  )

  logger.info("Saving csv...")
  df.to_csv(working_directory / SELECTED_FILENAME, sep=SEP, header=True, index=False)
  logger.info("Saving txt...")
  (working_directory / SELECTED_TXT_FILENAME).write_text(df_to_txt(df))
  logger.info("Saving consecutive txt...")
  (working_directory / SELECTED_TXT_CONSEC_FILENAME).write_text(df_to_consecutive_txt(df))
  logger.info("Saving tex...")
  (working_directory / SELECTED_TEX_FILENAME).write_text(df_to_tex(df))
  logger.info("Saving paths...")
  (working_directory / SELECTED_PATHS_FILENAME).write_text(df_to_paths(df))
  logger.info(f"Done. Saved script to: {working_directory / SELECTED_TXT_FILENAME}")


def init_app_generate_deselected_script_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('--sorting_mode', choices=SortingMode,
                      type=SortingMode.__getitem__, default=SortingMode.BY_INDEX)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--parts-count', type=int, default=None)
  parser.add_argument('--take-per-part', type=int, default=None)
  parser.add_argument('--ignore-symbols', type=str, nargs="*", default=[])
  parser.add_argument('--only-txt', action='store_true')
  return app_generate_deselected_script


def app_generate_deselected_script(working_directory: Path, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None, only_txt: bool = True) -> None:
  logger = getLogger(__name__)
  if not working_directory.exists():
    logger.info("Corpus does not exist.")
    return

  selection, reading_passages, representations, paths, reading_passages_paths = load_data(
    working_directory)
  logger.info("Saving deselected script...")
  deselected = representations.keys() - selection
  df = get_df(
    reading_passages=reading_passages,
    representations=representations,
    selection=deselected,
    paths=paths,
    read_paths=reading_passages_paths,
    mode=sorting_mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    parts_count=parts_count,
    take_per_part=take_per_part,
  )

  logger.info("Getting txt...")
  txt = df_to_txt(df)
  logger.info("Saving txt...")
  (working_directory / DESELECTED_TXT_FILENAME).write_text(txt)
  logger.info(f"Done. Saved script to: {working_directory / DESELECTED_TXT_FILENAME}")

  if not only_txt:
    logger.info("Saving csv...")
    df.to_csv(working_directory / DESELECTED_FILENAME, sep=SEP, header=True, index=False)
    logger.info("Saving tex...")
    (working_directory / DESELECTED_TEX_FILENAME).write_text(df_to_tex(df))
    logger.info("Saving consecutive txt...")
    (working_directory / DESELECTED_TXT_CONSEC_FILENAME).write_text(df_to_consecutive_txt(df))
    logger.info("Saving paths...")
    (working_directory / DESELECTED_PATHS_FILENAME).write_text(df_to_paths(df))


def init_generate_textgrid_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('--reading-speed-chars-per-s', type=float, default=DEFAULT_AVG_CHARS_PER_S)
  return app_generate_textgrid


def app_generate_textgrid(working_directory: Path, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting from tex...")
  tex_path = get_tex_path(working_directory)
  assert tex_path.exists()
  tex_content = tex_path.read_text()

  reading_passages = load_reading_passages(working_directory)
  representations = load_representations(working_directory)

  grid = generate_textgrid(
    reading_passages=reading_passages,
    representations=representations,
    tex=tex_content,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  grid_path = working_directory / "textgrid.TextGrid"
  grid.write(grid_path)

  logger.info("Done.")
