import os
from enum import IntEnum
from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Optional, Set, Tuple

from general_utils import load_obj, save_obj
from general_utils.main import (get_all_files_in_all_subfolders, get_filepaths,
                                get_subfolders)
from recording_script_generator.core.export import (SortingMode,
                                                    df_to_consecutive_txt,
                                                    df_to_tex, df_to_txt,
                                                    generate_textgrid,
                                                    get_reading_scripts)
from recording_script_generator.core.main import (
    Metadata, Passages, ReadingPassages, Representations,
    add_corpus_from_text_files, add_corpus_from_texts, change_ipa, change_text,
    convert_eng_to_arpa, deselect_all_utterances, map_to_ipa, merge, normalize,
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
from text_selection.selection import SelectionMode
from text_utils import Language
from text_utils.pronunciation.main import EngToIPAMode
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Symbol
from tqdm import tqdm

READING_PASSAGES_DATA_FILE = "reading_passages.pkl"
REPRESENTATIONS_DATA_FILE = "representations.pkl"
METADATA_DATA_FILE = "metadata.pkl"
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
REST_TEX_CONSEC_FILENAME = "rest_consecutive.tex"


class PreparationTarget(IntEnum):
  READING_PASSAGES = 0
  REPRESENTATIONS = 1
  BOTH = 2


def get_corpus_dir(base_dir: Path, corpus_name: str) -> Path:
  return base_dir / corpus_name


def get_step_dir(corpus_dir: Path, step_name: str) -> Path:
  return corpus_dir / step_name


def load_reading_passages(step_dir: Path) -> ReadingPassages:
  res = load_obj(step_dir / READING_PASSAGES_DATA_FILE)
  return res


def load_representations(step_dir: Path) -> Representations:
  res = load_obj(step_dir / REPRESENTATIONS_DATA_FILE)
  return res


def load_metadata(step_dir: Path) -> Metadata:
  res = load_obj(step_dir / METADATA_DATA_FILE)
  return res


def save_reading_passages(step_dir: Path, reading_passages: ReadingPassages) -> None:
  logger = getLogger(__name__)
  logger.info("Saving reading passages...")
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / READING_PASSAGES_DATA_FILE,
    obj=reading_passages,
  )
  logger.info("Done.")

  return None


def save_representations(step_dir: Path, representations: Representations) -> None:
  logger = getLogger(__name__)
  logger.info("Saving representations...")
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / REPRESENTATIONS_DATA_FILE,
    obj=representations,
  )
  logger.info("Done.")

  return None


def save_metadata(step_dir: Path, metadata: Metadata) -> None:
  logger = getLogger(__name__)
  logger.info("Saving metadata...")
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / METADATA_DATA_FILE,
    obj=metadata,
  )
  logger.info("Done.")

  return None


def _save_stats_df(step_dir: Path, metadata: Metadata, data: Passages) -> None:
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


def _save_scripts(step_dir: Path, metadata: Metadata, data: Passages, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[str]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
  logger = getLogger(__name__)
  logger.info("Saving scripts...")

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
  logger.info("Done.")


def get_tex_path(step_dir: Path) -> Path:
  return step_dir / SELECTED_TEX_FILENAME


def app_add_corpus_from_text_file(base_dir: Path, corpus_name: str, step_name: str, text_path: Path, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  if not text_path.exists():
    logger = getLogger(__name__)
    logger.error("File does not exist.")
    return

  text = text_path.read_text()
  app_add_corpus_from_text(base_dir, corpus_name, step_name, text, lang, text_format, overwrite)


def app_add_corpus_from_text_files(base_dir: Path, corpus_name: str, step_name: str, text_dir: Path, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
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

  reading_passages, representations, metadata = add_corpus_from_text_files(
    files=all_txt_files,
    lang=lang,
    text_format=text_format,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Done.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_metadata(step_dir, metadata)
  save_reading_passages(step_dir, reading_passages)
  save_representations(step_dir, representations)


def app_add_corpus_from_text(base_dir: Path, corpus_name: str, step_name: str, text: str, lang: Language, text_format: SymbolFormat, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)

  if corpus_dir.exists() and not overwrite:
    logger.info("Corpus already exists.")
    return

  reading_passages, representations, metadata = add_corpus_from_texts(
    text=text,
    lang=lang,
    text_format=text_format,
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_metadata(step_dir, metadata)
  save_reading_passages(step_dir, reading_passages)
  save_representations(step_dir, representations)


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
  assert corpora_step_names is not None

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


def _alter_data(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str], overwrite: bool, method: Callable[[Metadata, Passages], None]):
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

  targets = []
  if target == PreparationTarget.BOTH:
    targets.append(PreparationTarget.READING_PASSAGES)
    targets.append(PreparationTarget.REPRESENTATIONS)
  else:
    targets.append(target)

  for i, prep_target in enumerate(targets):
    metadata = load_metadata(in_step_dir)
    if prep_target == PreparationTarget.READING_PASSAGES:
      data = load_reading_passages(in_step_dir)
    elif prep_target == PreparationTarget.REPRESENTATIONS:
      data = load_representations(in_step_dir)
    else:
      assert False

    method(metadata, data)

    on_first_prep_target = i == 0
    if on_first_prep_target and out_step_dir.exists():
      assert overwrite
      logger.info("Removing existing out dir...")
      rmtree(out_step_dir)
      logger.info("Done.")
    out_step_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(out_step_dir, metadata)
    if prep_target == PreparationTarget.READING_PASSAGES:
      save_reading_passages(out_step_dir, data)
    elif prep_target == PreparationTarget.REPRESENTATIONS:
      save_representations(out_step_dir, data)
    else:
      assert False


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  method = partial(
    normalize,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_convert_to_arpa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to ARPA...")
  method = partial(
    convert_eng_to_arpa,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_map_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Mapping to IPA...")
  method = partial(
    map_to_ipa,
    target=target,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_change_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_ipa,
    target=target,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
  )

  _alter_data(base_dir, corpus_name, in_step_name, out_step_name, overwrite, method)


def app_change_text(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, remove_space_around_punctuation: bool, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_text,
    target=target,
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
  tex_content = tex_path.read_text()
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
  _alter_data(base_dir, corpus_name, in_step_name, target,
              out_step_name, overwrite, remove_utterances_with_acronyms)


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
  tex_content = tex_path.read_text()

  data = load_corpus(step_dir)

  grid = generate_textgrid(
    data=data,
    tex=tex_content,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
  )

  grid_path = step_dir / "textgrid.TextGrid"
  grid.write(grid_path)

  logger.info("Done.")
