from enum import IntEnum
from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Optional, Set, Tuple

from general_utils import load_obj, save_obj
from general_utils.main import get_all_files_in_all_subfolders
from recording_script_generator.core.export import (SortingMode,
                                                    df_to_consecutive_txt,
                                                    df_to_tex, df_to_txt,
                                                    generate_textgrid,
                                                    get_reading_scripts)
from recording_script_generator.core.main import (
    ReadingPassages, Representations, Selection, Utterances,
    add_corpus_from_text_files, add_corpus_from_texts, change_ipa, change_text,
    convert_eng_passages_to_arpa, deselect_all_utterances,
    get_utterance_durations_based_on_symbols, map_passages_to_ipa, merge,
    normalize, remove_deselected, remove_duplicate_utterances,
    remove_non_existent_utterances, remove_utterances_with_acronyms,
    remove_utterances_with_proper_names,
    remove_utterances_with_too_seldom_words,
    remove_utterances_with_undesired_sentence_lengths,
    remove_utterances_with_undesired_text,
    remove_utterances_with_unknown_words, select_all_utterances,
    select_from_tex, select_greedy_ngrams_duration,
    select_greedy_ngrams_epochs, select_kld_ngrams_duration)
from recording_script_generator.core.stats import (get_n_gram_stats_df,
                                                   log_general_stats)
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_IGNORE, DEFAULT_SEED,
                                                DEFAULT_SPLIT_BOUNDARY_MAX_S,
                                                DEFAULT_SPLIT_BOUNDARY_MIN_S,
                                                SEP)
from text_selection.selection import SelectionMode
from text_utils import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Symbol

READING_PASSAGES_DATA_FILE = "reading_passages.pkl"
REPRESENTATIONS_DATA_FILE = "representations.pkl"
SELECTION_DATA_FILE = "selection.pkl"
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


def load_selection(step_dir: Path) -> Selection:
  res = load_obj(step_dir / SELECTION_DATA_FILE)
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


def save_selection(step_dir: Path, selection: Selection) -> None:
  logger = getLogger(__name__)
  logger.info("Saving selection...")
  step_dir.mkdir(parents=True, exist_ok=True)
  save_obj(
    path=step_dir / SELECTION_DATA_FILE,
    obj=selection,
  )
  logger.info("Done.")

  return None


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


def _save_scripts(step_dir: Path, reading_passages: ReadingPassages, representations: Representations, selection: Selection, sorting_mode: SortingMode, seed: Optional[int] = DEFAULT_SEED, ignore_symbols: Optional[Set[str]] = DEFAULT_IGNORE, parts_count: Optional[int] = None, take_per_part: Optional[int] = None) -> None:
  logger = getLogger(__name__)
  logger.info("Saving scripts...")

  selected_df, rest_df = get_reading_scripts(
    reading_passages=reading_passages,
    representations=representations,
    selection=selection,
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

  selection, reading_passages, representations = add_corpus_from_text_files(
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
  save_selection(step_dir, selection)
  save_reading_passages(step_dir, reading_passages)
  save_representations(step_dir, representations)


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
  )

  if corpus_dir.exists():
    assert overwrite
    logger.info("Removing existing corpus...")
    rmtree(corpus_dir)
    logger.info("Removed existing corpus.")

  step_dir = get_step_dir(corpus_dir, step_name)
  save_selection(step_dir, selection)
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

  selection = load_selection(step_dir)
  reading_passages = load_reading_passages(step_dir)
  representations = load_representations(step_dir)

  _save_scripts(
    reading_passages=reading_passages,
    representations=representations,
    selection=selection,
    step_dir=step_dir,
    sorting_mode=sorting_mode,
    seed=seed,
    ignore_symbols=ignore_symbols,
    parts_count=parts_count,
    take_per_part=take_per_part,
  )


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

    selection = load_selection(in_step_dir)
    reading_passages = load_reading_passages(in_step_dir)
    representations = load_representations(in_step_dir)

    data_to_merge.append((selection, reading_passages, representations))

  merged_selection, merged_reading_passages, merged_representations = merge(data_to_merge)

  if out_corpus_dir.exists():
    assert overwrite
    rmtree(out_corpus_dir)
    logger.info("Removed existing corpus.")

  out_corpus_dir.mkdir(parents=True, exist_ok=False)
  out_step_dir = get_step_dir(out_corpus_dir, out_step_name)
  save_selection(out_step_dir, merged_selection)
  save_reading_passages(out_step_dir, merged_reading_passages)
  save_representations(out_step_dir, merged_representations)


def __alter_data(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str], overwrite: bool, method: Callable[[Utterances, Selection], None]):
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
    selection = load_selection(in_step_dir)
    if prep_target == PreparationTarget.READING_PASSAGES:
      utterances = load_reading_passages(in_step_dir)
    elif prep_target == PreparationTarget.REPRESENTATIONS:
      utterances = load_representations(in_step_dir)
    else:
      assert False

    method(utterances, selection)

    on_first_prep_target = i == 0
    if on_first_prep_target and out_step_dir.exists():
      assert overwrite
      logger.info("Removing existing out dir...")
      rmtree(out_step_dir)
      logger.info("Done.")
    out_step_dir.mkdir(parents=True, exist_ok=True)

    save_selection(out_step_dir, selection)
    if prep_target == PreparationTarget.READING_PASSAGES:
      save_reading_passages(out_step_dir, utterances)
    elif prep_target == PreparationTarget.REPRESENTATIONS:
      save_representations(out_step_dir, utterances)
    else:
      assert False

    if target == PreparationTarget.BOTH:
      pass
    elif target == PreparationTarget.REPRESENTATIONS:
      logger.info("Updating reading passages...")
      reading_passages = load_reading_passages(in_step_dir)
      remove_non_existent_utterances(utterances, reading_passages)
      save_reading_passages(out_step_dir, reading_passages)
      logger.info("Done.")
    elif target == PreparationTarget.READING_PASSAGES:
      logger.info("Updating representations...")
      representations = load_representations(in_step_dir)
      remove_non_existent_utterances(utterances, representations)
      save_representations(out_step_dir, representations)
      logger.info("Done.")
    else:
      assert False


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, normalize)


def app_convert_to_arpa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Converting to ARPA...")
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, convert_eng_passages_to_arpa)


def app_map_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Mapping to IPA...")
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, map_passages_to_ipa)


def app_change_ipa(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_change_text(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, remove_space_around_punctuation: bool, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_text,
    remove_space_around_punctuation=remove_space_around_punctuation,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_select_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting all...")
  target = PreparationTarget.REPRESENTATIONS
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, select_all_utterances)


def app_deselect_all(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Deselecting all...")
  target = PreparationTarget.REPRESENTATIONS
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, deselect_all_utterances)


def app_select_from_tex(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
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

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_remove_deselected(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing deselected...")
  target = PreparationTarget.REPRESENTATIONS
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, remove_deselected)


def app_remove_undesired_text(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, undesired: Set[Symbol], out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing undesired text...")
  method = partial(
    remove_utterances_with_undesired_text,
    undesired=undesired,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_remove_duplicate_utterances(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing duplicate utterances...")
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, remove_duplicate_utterances)


def app_remove_utterances_with_proper_names(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with proper names...")
  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name,
               overwrite, remove_utterances_with_proper_names)


def app_remove_utterances_with_acronyms(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with acronyms...")
  __alter_data(base_dir, corpus_name, in_step_name, target,
               out_step_name, overwrite, remove_utterances_with_acronyms)


def app_remove_utterances_with_undesired_sentence_lengths(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, min_word_count: Optional[int] = None, max_word_count: Optional[int] = None, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with undesired word counts...")
  method = partial(
    remove_utterances_with_undesired_sentence_lengths,
    min_word_count=min_word_count,
    max_word_count=max_word_count,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_remove_utterances_with_unknown_words(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, max_unknown_word_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with to many unknown words...")
  method = partial(
    remove_utterances_with_unknown_words,
    max_unknown_word_count=max_unknown_word_count,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_remove_utterances_with_too_seldom_words(base_dir: Path, corpus_name: str, in_step_name: str, target: PreparationTarget, min_occurrence_count: int, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Removing utterances with too seldom words...")
  method = partial(
    remove_utterances_with_too_seldom_words,
    min_occurrence_count=min_occurrence_count,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_select_greedy_ngrams_epochs(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]], target: PreparationTarget, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy...")
  method = partial(
    select_greedy_ngrams_epochs,
    n_gram=n_gram,
    epochs=epochs,
    ignore_symbols=ignore_symbols,
  )

  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


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


def app_select_greedy_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, ignore_symbols: Optional[Set[Symbol]], reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with Greedy...")
  
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
  reading_passages = load_reading_passages(in_step_dir)
  utterance_durations_s = get_utterance_durations_based_on_symbols(reading_passages, reading_speed_chars_per_s)

  method = partial(
    select_greedy_ngrams_duration,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    minutes=minutes,
    reading_speed_chars_per_s=reading_speed_chars_per_s,
    utterance_durations_s=utterance_durations_s,
    mode=SelectionMode.SHORTEST,
  )
  
  target = PreparationTarget.REPRESENTATIONS
  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


def app_select_kld_ngrams_duration(base_dir: Path, corpus_name: str, in_step_name: str, n_gram: int, minutes: float, reading_speed_chars_per_s: float = DEFAULT_AVG_CHARS_PER_S, ignore_symbols: Set[Symbol] = DEFAULT_IGNORE, boundary_min_s: float = DEFAULT_SPLIT_BOUNDARY_MIN_S, boundary_max_s: float = DEFAULT_SPLIT_BOUNDARY_MAX_S, out_step_name: Optional[str] = None, overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info("Selecting utterances with KLD...")
  
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
  reading_passages = load_reading_passages(in_step_dir)
  utterance_durations_s = get_utterance_durations_based_on_symbols(reading_passages, reading_speed_chars_per_s)

  method = partial(
    select_kld_ngrams_duration,
    n_gram=n_gram,
    minutes=minutes,
    ignore_symbols=ignore_symbols,
    utterance_durations_s=utterance_durations_s,
    boundary=(boundary_min_s, boundary_max_s),
  )

  target = PreparationTarget.REPRESENTATIONS
  __alter_data(base_dir, corpus_name, in_step_name, target, out_step_name, overwrite, method)


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
