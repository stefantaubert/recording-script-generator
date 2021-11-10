from logging import getLogger
from pathlib import Path
from typing import Optional, Set

from recording_script_generator.app.io import *
from recording_script_generator.core.exporting import (SortingMode,
                                                       df_to_consecutive_txt,
                                                       df_to_tex, df_to_txt,
                                                       generate_textgrid,
                                                       get_reading_script_df)
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_IGNORE, DEFAULT_SEED,
                                                SEP)
from text_utils.types import Symbol

# DATA_FILE = "data.pkl"
SELECTED_FILENAME = "selected.csv"
SELECTED_TXT_FILENAME = "selected.txt"
SELECTED_TXT_CONSEC_FILENAME = "selected_consecutive.txt"
SELECTED_TEX_FILENAME = "selected.tex"
ONE_GRAM_STATS_CSV_FILENAME = "1_gram_stats.csv"
TWO_GRAM_STATS_CSV_FILENAME = "2_gram_stats.csv"
THREE_GRAM_STATS_CSV_FILENAME = "3_gram_stats.csv"
REST_FILENAME = "rest.csv"
REST_TXT_FILENAME = "rest.txt"
REST_TEX_FILENAME = "rest.tex"
REST_TEX_FILENAME = "rest.tex"
REST_TXT_CONSEC_FILENAME = "rest_consecutive.tex"


def get_tex_path(step_dir: Path) -> Path:
  return step_dir / SELECTED_TEX_FILENAME


def app_generate_selected_script(base_dir: Path, corpus_name: str, step_name: str, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[Symbol]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if not corpus_dir.exists():
    logger.info("Corpus does not exist.")
    return

  step_dir = get_step_dir(corpus_dir, step_name)

  if not step_dir.exists():
    logger.info("Step does not exist.")
    return

  selection, reading_passages, representations = load_data(step_dir)
  logger.info("Saving selected scripts...")

  selected_df = get_reading_script_df(
    reading_passages=reading_passages,
    representations=representations,
    selection=selection,
    mode=sorting_mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    parts_count=parts_count,
    take_per_part=take_per_part,
  )

  logger.info("Saving csv...")
  selected_df.to_csv(step_dir / SELECTED_FILENAME, sep=SEP, header=True, index=False)
  logger.info("Saving txt...")
  (step_dir / SELECTED_TXT_FILENAME).write_text(df_to_txt(selected_df))
  logger.info("Saving consecutive txt...")
  (step_dir / SELECTED_TXT_CONSEC_FILENAME).write_text(df_to_consecutive_txt(selected_df))
  logger.info("Saving tex...")
  get_tex_path(step_dir).write_text(df_to_tex(selected_df))
  logger.info(f"Done. Saved script to: {step_dir / SELECTED_TXT_FILENAME}")


def app_generate_deselected_script(base_dir: Path, corpus_name: str, step_name: str, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[Symbol]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None, only_txt: bool = True) -> None:
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if not corpus_dir.exists():
    logger.info("Corpus does not exist.")
    return

  step_dir = get_step_dir(corpus_dir, step_name)

  if not step_dir.exists():
    logger.info("Step does not exist.")
    return

  selection, reading_passages, representations = load_data(step_dir)
  logger.info("Saving deselected script...")
  deselected = representations.keys() - selection
  rest_df = get_reading_script_df(
    reading_passages=reading_passages,
    representations=representations,
    selection=deselected,
    mode=sorting_mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    parts_count=parts_count,
    take_per_part=take_per_part,
  )

  logger.info("Saving tex...")
  (step_dir / REST_TEX_FILENAME).write_text(df_to_tex(rest_df))
  logger.info(f"Done. Saved script to: {step_dir / REST_TXT_FILENAME}")

  if not only_txt:
    logger.info("Saving csv...")
    rest_df.to_csv(step_dir / REST_FILENAME, sep=SEP, header=True, index=False)
    logger.info("Saving txt...")
    (step_dir / REST_TXT_FILENAME).write_text(df_to_txt(rest_df))
    logger.info("Saving consecutive txt...")
    (step_dir / REST_TXT_CONSEC_FILENAME).write_text(df_to_consecutive_txt(rest_df))


def app_generate_textgrid(base_dir: Path, corpus_name: str, step_name: str, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting from tex...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  step_dir = get_step_dir(corpus_dir, step_name)
  tex_path = get_tex_path(step_dir)
  assert tex_path.exists()
  tex_content = tex_path.read_text()

  reading_passages = load_reading_passages(step_dir)
  representations = load_representations(step_dir)

  grid = generate_textgrid(
    reading_passages=reading_passages,
    representations=representations,
    tex=tex_content,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  grid_path = step_dir / "textgrid.TextGrid"
  grid.write(grid_path)

  logger.info("Done.")
