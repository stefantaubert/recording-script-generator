from functools import partial
from logging import getLogger
from math import inf
from pathlib import Path
from shutil import rmtree
from typing import Callable, Dict, List, Optional, Set, Tuple

from pandas import DataFrame
from recording_script_generator.core.export import (SortingMode, df_to_tex,
                                                    df_to_txt,
                                                    get_reading_scripts)
from recording_script_generator.core.main import (
    PreparationData, PreparationTarget, SplitBoundary, add_corpus_from_text,
    boundaries_are_distinct, convert_to_ipa, merge, normalize,
    remove_duplicate_utterances, remove_utterances_with_acronyms,
    remove_utterances_with_proper_names,
    remove_utterances_with_too_seldom_words,
    remove_utterances_with_undesired_sentence_lengths,
    remove_utterances_with_undesired_text,
    remove_utterances_with_unknown_words, select_all_utterances,
    select_greedy_ngrams_duration, select_greedy_ngrams_epochs,
    select_kld_ngrams_duration, select_kld_ngrams_duration_split,
    select_kld_ngrams_iterations)
from recording_script_generator.core.stats import (get_n_gram_stats_df,
                                                   log_general_stats)
from recording_script_generator.utils import (get_subdir, load_obj, read_lines,
                                              read_text, save_json, save_obj)
from text_selection.selection import SelectionMode
from text_utils import Language
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import EngToIpaMode

DEFAULT_SEED = 1111
DEFAULT_SORTING_MODE = SortingMode.BY_INDEX
AVG_CHARS_PER_S = 25
DATA_FILE = "data.pkl"
SELECTED_FILENAME = "selected.csv"
SELECTED_TXT_FILENAME = "selected.txt"
SELECTED_TEX_FILENAME = "selected.tex"
ONE_GRAM_STATS_CSV_FILENAME = "1_gram_stats.csv"
TWO_GRAM_STATS_CSV_FILENAME = "2_gram_stats.csv"
THREE_GRAM_STATS_CSV_FILENAME = "3_gram_stats.csv"
REST_FILENAME = "rest.csv"
REST_TXT_FILENAME = "rest.txt"
REST_TEX_FILENAME = "rest.tex"
SEP = "\t"
DEFAULT_IGNORE = {}
DEFAULT_SPLIT_BOUNDARY = (0, inf)


def get_corpus_dir(base_dir: Path, corpus_name: str) -> Path:
  return base_dir / corpus_name


def get_step_dir(corpus_dir: Path, step_name: str) -> Path:
  return corpus_dir / step_name


def load_corpus(step_dir: Path) -> PreparationData:
  res = load_obj(step_dir / DATA_FILE)
  return res


def save_corpus(step_dir: Path, corpus: PreparationData) -> None:
  save_obj(
    path=step_dir / DATA_FILE,
    obj=corpus,
  )
  return None


def _save_stats_df(step_dir: Path, data: PreparationData) -> None:
  logger = getLogger(__name__)
  logger.info("Getting 1-gram stats...")
  one_gram_repr_stats = get_n_gram_stats_df(data.representations, data.selected, n=1)
  one_gram_repr_stats.to_csv(step_dir / ONE_GRAM_STATS_CSV_FILENAME,
                             sep=SEP, header=True, index=False)

  logger.info("Getting 2-gram stats...")
  two_gram_repr_stats = get_n_gram_stats_df(data.representations, data.selected, n=2)
  two_gram_repr_stats.to_csv(step_dir / TWO_GRAM_STATS_CSV_FILENAME,
                             sep=SEP, header=True, index=False)

  logger.info("Getting 3-gram stats...")
  three_gram_repr_stats = get_n_gram_stats_df(data.representations, data.selected, n=3)
  three_gram_repr_stats.to_csv(step_dir / THREE_GRAM_STATS_CSV_FILENAME,
                               sep=SEP, header=True, index=False)
  logger.info("Done.")


def save_scripts(step_dir: Path, data: PreparationData, sorting_mode: SortingMode) -> None:
  selected_df, rest_df = get_reading_scripts(
    data, mode=sorting_mode, seed=DEFAULT_SEED)
  selected_df.to_csv(step_dir / SELECTED_FILENAME, sep=SEP, header=True, index=False)
  rest_df.to_csv(step_dir / REST_FILENAME, sep=SEP, header=True, index=False)
  (step_dir / SELECTED_TXT_FILENAME).write_text(df_to_txt(selected_df))
  (step_dir / SELECTED_TEX_FILENAME).write_text(df_to_tex(selected_df))
  (step_dir / REST_TXT_FILENAME).write_text(df_to_txt(rest_df))
  (step_dir / REST_TEX_FILENAME).write_text(df_to_tex(rest_df))


def app_add_corpus_from_text_file(base_dir: Path, corpus_name: str, step_name: str, text_path: Path, lang: Language, ignore_tones: Optional[bool] = False, ignore_arcs: Optional[bool] = True, replace_unknown_ipa_by: Optional[str] = "_", overwrite: bool = False) -> None:
  if not text_path.exists():
    logger = getLogger(__name__)
    logger.error("File does not exist.")
    return

  text = read_text(text_path)
  app_add_corpus_from_text(base_dir, corpus_name, step_name, text, lang,
                           ignore_tones, ignore_arcs, replace_unknown_ipa_by, overwrite)


def log_stats(base_dir: Path, corpus_name: str, step_name: str) -> None:
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if not corpus_dir.exists():
    logger.info("Corpus does not exist.")
    return

  step_dir = get_step_dir(corpus_dir, step_name)

  if not step_dir.exists():
    logger.info("Step does not exist.")
    return

  data = load_corpus(step_dir)

  _log_stats(data, step_dir)


def _log_stats(data: PreparationData, step_dir: Path):
  log_general_stats(data, AVG_CHARS_PER_S)
  _save_stats_df(step_dir, data)


def app_add_corpus_from_text(base_dir: Path, corpus_name: str, step_name: str, text: str, lang: Language, ignore_tones: Optional[bool] = False, ignore_arcs: Optional[bool] = True, replace_unknown_ipa_by: Optional[str] = "_", overwrite: bool = False, sorting_mode: SortingMode = DEFAULT_SORTING_MODE) -> None:
  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  result = add_corpus_from_text(
    text=text,
    lang=lang,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
  )

  if corpus_dir.exists():
    assert overwrite
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  corpus_dir.mkdir(parents=True, exist_ok=False)
  step_dir = get_step_dir(corpus_dir, step_name)
  step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(step_dir, result)
  save_scripts(step_dir, result, sorting_mode)
  _log_stats(result, step_dir)


def app_merge_merged(base_dir: Path, corpora_step_names: List[Tuple[str, str]], out_corpus_name: str, out_step_name: str, overwrite: bool = False, sorting_mode: SortingMode = DEFAULT_SORTING_MODE):
  logger = getLogger(__name__)
  logger.info(f"Merging multiple corpora...")

  out_corpus_dir = get_corpus_dir(base_dir, out_corpus_name)
  if out_corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  data_to_merge = []
  for corpus_name, in_step_name in corpora_step_names:
    in_corpus_dir = get_corpus_dir(base_dir, corpus_name)
    in_step_dir = get_step_dir(in_corpus_dir, in_step_name)

    if not in_step_dir.exists():
      logger.error("In step dir does not exist!")
      return

    data = load_corpus(in_step_dir)
    data_to_merge.append(data)

  merged_data = merge(data_to_merge)

  if out_corpus_dir.exists():
    assert overwrite
    rmtree(out_corpus_dir)
    logger.info("Removed existing corpus.")

  out_corpus_dir.mkdir(parents=True, exist_ok=False)
  out_step_dir = get_step_dir(out_corpus_dir, out_step_name)
  out_step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(out_step_dir, merged_data)
  save_scripts(out_step_dir, merged_data, sorting_mode)
  _log_stats(merged_data, out_step_dir)


def _alter_data(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str], overwrite: bool, method: Callable[[PreparationData], None], sorting_mode: SortingMode = DEFAULT_SORTING_MODE):
  logger = getLogger(__name__)
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
  if out_step_name is None:
    out_step_dir = in_step_dir
  else:
    out_step_dir = get_step_dir(corpus_dir, out_step_name)

  if not corpus_dir.exists():
    logger.info("Corpus dir does not exists.")
    return
  if not in_step_dir.exists():
    logger.info("In dir not exists.")
    return
  if out_step_dir.exists() and not overwrite:
    logger.info("Already exists.")
    return

  data = load_corpus(in_step_dir)

  method(data)

  if out_step_dir.exists():
    assert overwrite
    rmtree(out_step_dir)
    logger.info("Overwriting existing out dir.")
  out_step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(out_step_dir, data)
  save_scripts(out_step_dir, data, sorting_mode)
  _log_stats(data, out_step_dir)


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, ignore_tones: Optional[bool] = False, ignore_arcs: Optional[bool] = True, replace_unknown_ipa_by: Optional[str] = "_", out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize,
    target=target,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_convert_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, mode: Optional[EngToIpaMode], ignore_tones: Optional[bool] = False, ignore_arcs: Optional[bool] = True, replace_unknown_ipa_by: Optional[str] = "_", replace_unknown_with: Optional[str] = "_", consider_ipa_annotations: bool = False, use_cache: Optional[bool] = True, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to IPA...")
  method = partial(
    convert_to_ipa,
    target=target,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
    mode=mode,
    replace_unknown_with=replace_unknown_with,
    consider_ipa_annotations=consider_ipa_annotations,
    use_cache=use_cache,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting all...")
  method = partial(
    select_all_utterances,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_undesired_text(base_dir: Path, corpus_name: str, in_step_name: str, undesired: Set[str], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing undesired text...")
  method = partial(
    remove_utterances_with_undesired_text,
    undesired=undesired,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_duplicate_utterances(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing duplicate utterances...")
  method = partial(
    remove_duplicate_utterances,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_proper_names(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with proper names...")
  method = partial(
    remove_utterances_with_proper_names,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_acronyms(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with acronyms...")
  method = partial(
    remove_utterances_with_acronyms,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_undesired_sentence_lengths(base_dir: Path, corpus_name: str, in_step_name: str, min_word_count: Optional[int] = None, max_word_count: Optional[int] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with undesired word counts...")
  method = partial(
    remove_utterances_with_undesired_sentence_lengths,
    min_word_count=min_word_count,
    max_word_count=max_word_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_unknown_words(base_dir: Path, corpus_name: str, in_step_name: str, max_unknown_word_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with to many unknown words...")
  method = partial(
    remove_utterances_with_unknown_words,
    max_unknown_word_count=max_unknown_word_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_too_seldom_words(base_dir: Path, corpus_name: str, in_step_name: str, min_occurrence_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with too seldom words...")
  method = partial(
    remove_utterances_with_too_seldom_words,
    min_occurrence_count=min_occurrence_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_kld_ngrams_iterations(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, iterations: int, ignore_symbols: Optional[Set[str]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD...")
  method = partial(
    select_kld_ngrams_iterations,
    n_gram=n_gram,
    iterations=iterations,
    ignore_symbols=ignore_symbols,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_kld_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: int = AVG_CHARS_PER_S, ignore_symbols: Set[str] = DEFAULT_IGNORE, boundary: SplitBoundary = DEFAULT_SPLIT_BOUNDARY, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD...")
  method = partial(
    select_kld_ngrams_duration,
    n_gram=n_gram,
    minutes=minutes,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    ignore_symbols=ignore_symbols,
    boundary=boundary,
    mode=SelectionMode.FIRST,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_kld_ngrams_duration_split(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, split_seconds_percent: Dict[SplitBoundary, float], reading_speed_chars_per_s: int = AVG_CHARS_PER_S, ignore_symbols: Optional[Set[str]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD...")

  if not boundaries_are_distinct(list(split_seconds_percent.keys())):
    logger.error("Split boundaries were not distinct!")
    return

  sum_percent = sum(split_seconds_percent.values())
  if not 0.9999 <= sum_percent <= 1.0001:
    logger.error(f"The percentages need to sum up to 100% (1.0), instead it was: {sum_percent}!")
    return

  method = partial(
    select_kld_ngrams_duration_split,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    split_seconds_percent=split_seconds_percent,
    mode=SelectionMode.FIRST,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_greedy_ngrams_epochs(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, epochs: int, ignore_symbols: Optional[Set[str]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy...")
  method = partial(
    select_greedy_ngrams_epochs,
    n_gram=n_gram,
    epochs=epochs,
    ignore_symbols=ignore_symbols,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_greedy_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: int = AVG_CHARS_PER_S, ignore_symbols: Optional[Set[str]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy...")
  method = partial(
    select_greedy_ngrams_duration,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    mode=SelectionMode.SHORTEST,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)
