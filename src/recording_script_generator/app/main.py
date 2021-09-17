from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Optional, Set, Tuple

from recording_script_generator.core.export import (SortingMode,
                                                    df_to_consecutive_txt,
                                                    df_to_tex, df_to_txt,
                                                    generate_textgrid,
                                                    get_reading_scripts)
from recording_script_generator.core.main import (
    PreparationData, PreparationTarget, add_corpus_from_text, change_ipa,
    convert_to_ipa, deselect_all_utterances, merge, normalize,
    remove_deselected, remove_duplicate_utterances,
    remove_utterances_with_acronyms, remove_utterances_with_proper_names,
    remove_utterances_with_too_seldom_words,
    remove_utterances_with_undesired_sentence_lengths,
    remove_utterances_with_undesired_text,
    remove_utterances_with_unknown_words, select_all_utterances,
    select_from_tex, select_greedy_ngrams_duration,
    select_greedy_ngrams_epochs, select_kld_ngrams_duration,
    select_kld_ngrams_iterations)
from recording_script_generator.core.stats import (get_n_gram_stats_df,
                                                   log_general_stats)
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_IGNORE, DEFAULT_SEED,
                                                DEFAULT_SORTING_MODE,
                                                DEFAULT_SPLIT_BOUNDARY_MAX_S,
                                                DEFAULT_SPLIT_BOUNDARY_MIN_S,
                                                SEP)
from recording_script_generator.utils import load_obj, read_text, save_obj
from text_selection.selection import SelectionMode
from text_utils import Language
from text_utils.pronunciation.main import EngToIPAMode
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Symbol

DATA_FILE = "data.pkl"
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
REST_TEX_CONSEC_FILENAME = "rest_consecutive.tex"


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


def _save_scripts(step_dir: Path, data: PreparationData, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[str]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
  selected_df, rest_df = get_reading_scripts(
    data=data,
    mode=sorting_mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    parts_count=parts_count,
    take_per_part=take_per_part,
  )
  selected_df.to_csv(step_dir / SELECTED_FILENAME, sep=SEP, header=True, index=False)
  rest_df.to_csv(step_dir / REST_FILENAME, sep=SEP, header=True, index=False)
  (step_dir / SELECTED_TXT_FILENAME).write_text(df_to_txt(selected_df))
  (step_dir / SELECTED_TXT_CONSEC_FILENAME).write_text(df_to_consecutive_txt(selected_df))
  get_tex_path(step_dir).write_text(df_to_tex(selected_df))
  (step_dir / REST_TXT_FILENAME).write_text(df_to_txt(rest_df))
  (step_dir / REST_TEX_FILENAME).write_text(df_to_tex(rest_df))
  (step_dir / REST_TEX_CONSEC_FILENAME).write_text(df_to_consecutive_txt(rest_df))


def get_tex_path(step_dir: Path) -> Path:
  return step_dir / SELECTED_TEX_FILENAME


def app_add_corpus_from_text_file(base_dir: Path, corpus_name: str, step_name: str, text_path: Path, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  if not text_path.exists():
    logger = getLogger(__name__)
    logger.error("File does not exist.")
    return

  text = read_text(text_path)
  app_add_corpus_from_text(base_dir, corpus_name, step_name, text, lang, text_format, overwrite)


def app_add_corpus_from_text(base_dir: Path, corpus_name: str, step_name: str, text: str, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  result = add_corpus_from_text(
    text=text,
    lang=lang,
    text_format=text_format,
  )

  if corpus_dir.exists():
    assert overwrite
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  corpus_dir.mkdir(parents=True, exist_ok=False)
  step_dir = get_step_dir(corpus_dir, step_name)
  step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(step_dir, result)
  _save_scripts(step_dir, result, DEFAULT_SORTING_MODE)


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

  data = load_corpus(step_dir)

  log_general_stats(data, reading_speed_chars_per_s)
  _save_stats_df(step_dir, data)


def app_generate_scripts(base_dir: Path, corpus_name: str, step_name: str, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[Symbol]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
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

  _save_scripts(step_dir, data, sorting_mode, seed, ignore_symbols, parts_count, take_per_part)


def app_merge(base_dir: Path, corpora_step_names: List[Tuple[str, str]], out_corpus_name: str, out_step_name: str, overwrite: bool = False):
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
  _save_scripts(out_step_dir, merged_data, DEFAULT_SORTING_MODE)


def _alter_data(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str], overwrite: bool, method: Callable[[PreparationData], None]):
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
  _save_scripts(out_step_dir, data, DEFAULT_SORTING_MODE)


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_convert_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, mode: Optional[EngToIPAMode], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to IPA...")
  method = partial(
    convert_to_ipa,
    target=target,
    mode=mode,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_change_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, remove_space_around_punctuation: bool, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_ipa,
    target=target,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    remove_space_around_punctuation=remove_space_around_punctuation,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting all...")
  method = partial(
    select_all_utterances,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_deselect_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Deselecting all...")
  method = partial(
    deselect_all_utterances,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_from_tex(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting from tex...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  step_dir = get_step_dir(corpus_dir, in_step_name)
  tex_path = get_tex_path(step_dir)
  assert tex_path.exists()
  tex_content = read_text(tex_path)
  method = partial(
    select_from_tex,
    tex=tex_content,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_deselected(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing deselected...")
  method = partial(
    remove_deselected,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_undesired_text(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, undesired: Set[Symbol], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing undesired text...")
  method = partial(
    remove_utterances_with_undesired_text,
    target=target,
    undesired=undesired,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_duplicate_utterances(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing duplicate utterances...")
  method = partial(
    remove_duplicate_utterances,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_proper_names(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with proper names...")
  method = partial(
    remove_utterances_with_proper_names,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_acronyms(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with acronyms...")
  method = partial(
    remove_utterances_with_acronyms,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_undesired_sentence_lengths(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, min_word_count: Optional[int] = None, max_word_count: Optional[int] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with undesired word counts...")
  method = partial(
    remove_utterances_with_undesired_sentence_lengths,
    target=target,
    min_word_count=min_word_count,
    max_word_count=max_word_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_unknown_words(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, max_unknown_word_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with to many unknown words...")
  method = partial(
    remove_utterances_with_unknown_words,
    target=target,
    max_unknown_word_count=max_unknown_word_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_remove_utterances_with_too_seldom_words(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, min_occurrence_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with too seldom words...")
  method = partial(
    remove_utterances_with_too_seldom_words,
    target=target,
    min_occurrence_count=min_occurrence_count,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_greedy_ngrams_epochs(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy...")
  method = partial(
    select_greedy_ngrams_epochs,
    n_gram=n_gram,
    epochs=epochs,
    ignore_symbols=ignore_symbols,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


# def app_select_kld_ngrams_iterations(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, iterations: int, ignore_symbols: Optional[Set[str]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
#   logger = getLogger(__name__)
#   logger.info("Selecting utterances with KLD...")
#   method = partial(
#     select_kld_ngrams_iterations,
#     n_gram=n_gram,
#     iterations=iterations,
#     ignore_symbols=ignore_symbols,
#   )

#   _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_select_greedy_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, ignore_symbols: Optional[Set[Symbol]] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
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


def app_select_kld_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, boundary_min_s: float = DEFAULT_SPLIT_BOUNDARY_MIN_S, boundary_max_s: float = DEFAULT_SPLIT_BOUNDARY_MAX_S, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD...")
  method = partial(
    select_kld_ngrams_duration,
    n_gram=n_gram,
    minutes=minutes,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    ignore_symbols=ignore_symbols,
    boundary=(boundary_min_s, boundary_max_s),
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_generate_textgrid(base_dir: Path, corpus_name: str, step_name: str, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting from tex...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  step_dir = get_step_dir(corpus_dir, step_name)
  tex_path = get_tex_path(step_dir)
  assert tex_path.exists()
  tex_content = read_text(tex_path)

  data = load_corpus(step_dir)

  grid = generate_textgrid(
    data=data,
    tex=tex_content,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  grid_path = step_dir / "textgrid.TextGrid"
  grid.write(grid_path)

  logger.info("Done.")
