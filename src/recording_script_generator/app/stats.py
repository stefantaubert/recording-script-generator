from logging import getLogger
from pathlib import Path

from recording_script_generator.app.io import *
from recording_script_generator.core.stats import (get_n_gram_stats_df,
                                                   log_general_stats)
from recording_script_generator.core.types import Representations, Selection
from recording_script_generator.globals import DEFAULT_AVG_CHARS_PER_S, SEP

ONE_GRAM_STATS_CSV_FILENAME = "1_gram_stats.csv"
TWO_GRAM_STATS_CSV_FILENAME = "2_gram_stats.csv"
THREE_GRAM_STATS_CSV_FILENAME = "3_gram_stats.csv"


def _save_stats_df(step_dir: Path, selection: Selection, representations: Representations) -> None:
  logger = getLogger(__name__)
  logger.info("Getting 1-gram stats...")
  one_gram_repr_stats = get_n_gram_stats_df(representations, selection, n=1)
  one_gram_repr_stats.to_csv(step_dir / ONE_GRAM_STATS_CSV_FILENAME,
                             sep=SEP, header=True, index=False)

  logger.info("Getting 2-gram stats...")
  two_gram_repr_stats = get_n_gram_stats_df(representations, selection, n=2)
  two_gram_repr_stats.to_csv(step_dir / TWO_GRAM_STATS_CSV_FILENAME,
                             sep=SEP, header=True, index=False)

  logger.info("Getting 3-gram stats...")
  three_gram_repr_stats = get_n_gram_stats_df(representations, selection, n=3)
  three_gram_repr_stats.to_csv(step_dir / THREE_GRAM_STATS_CSV_FILENAME,
                               sep=SEP, header=True, index=False)
  logger.info("Done.")


def app_log_stats(base_dir: Path, corpus_name: str, step_name: str, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S) -> None:
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if not corpus_dir.exists():
    logger.info("Corpus does not exist.")
    return

  step_dir = get_step_dir(corpus_dir, step_name)

  if not step_dir.exists():
    logger.info("Step does not exist.")
    return

  selection = load_selection(step_dir)
  reading_passages = load_reading_passages(step_dir)
  representations = load_representations(step_dir)

  log_general_stats(
    reading_passages=reading_passages,
    representations=representations,
    selection=selection,
    avg_chars_per_s=reading_speed_chars_per_s,
  )

  _save_stats_df(step_dir, selection, representations)
