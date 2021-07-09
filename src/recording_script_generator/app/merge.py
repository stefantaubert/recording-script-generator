from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

from pandas.core.frame import DataFrame
from recording_script_generator.app.preparation import (get_corpus_dir,
                                                        get_step_dir,
                                                        load_corpus)
from recording_script_generator.core.merge import (ScriptData, Selection,
                                                   get_reading_scripts, ignore,
                                                   log_stats, merge,
                                                   merge_merged,
                                                   select_greedy_ngrams_epochs,
                                                   select_kld_ngrams_epochs,
                                                   select_rest)
from recording_script_generator.core.preparation import PreparationData
from recording_script_generator.utils import load_obj, save_obj

SCRIPTS_DIR_NAME = "scripts"
DATA_FILENAME = "data.pkl"
SELECTION_FILENAME = "selection.pkl"
SELECTED_FILENAME = "selected.csv"
SELECTED_TXT_FILENAME = "selected.txt"
IGNORED_FILENAME = "ignored.csv"
REST_FILENAME = "rest.csv"

SEP = "\t"


def get_merge_dir(base_dir: Path, merge_name: str) -> Path:
  return base_dir / SCRIPTS_DIR_NAME / merge_name


def get_script_dir(get_merge_dir: Path, script_name: str) -> Path:
  return get_merge_dir / script_name


def save_data(merge_dir: Path, data: ScriptData) -> None:
  path = merge_dir / DATA_FILENAME
  save_obj(path, data)


def load_data(merge_dir: Path) -> ScriptData:
  path = merge_dir / DATA_FILENAME
  res = load_obj(path)
  return res


def load_selection(script_dir: Path) -> Selection:
  path = script_dir / SELECTION_FILENAME
  res = load_obj(path)
  return res


def save_selection(script_dir: Path, selection: Selection) -> None:
  path = script_dir / SELECTION_FILENAME
  save_obj(path, selection)


def df_to_txt(df: DataFrame) -> str:
  result = ""
  for i, row in df.iterrows():
    result += f"{row['nr']}: {row['utterance']}\n"
  return result


def save_scripts(script_dir: Path, data: ScriptData, selection: Selection) -> None:
  selected_df, ignored_df, rest_df = get_reading_scripts(data, selection)
  selected_df.to_csv(script_dir / SELECTED_FILENAME, sep=SEP, header=True, index=False)
  ignored_df.to_csv(script_dir / IGNORED_FILENAME, sep=SEP, header=True, index=False)
  rest_df.to_csv(script_dir / REST_FILENAME, sep=SEP, header=True, index=False)
  selected_txt = df_to_txt(selected_df)
  (script_dir / SELECTED_TXT_FILENAME).write_text(selected_txt)


def app_merge(base_dir: Path, merge_name: str, script_name: str, corpora: List[Tuple[str, str]], overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Merging...")
  merge_dir = get_merge_dir(base_dir, merge_name)
  if merge_dir.exists():
    if overwrite:
      rmtree(merge_dir)
      logger.info("Removed existing merged data.")
    else:
      logger.info("Already exists.")
      return

  corpora_data: List[PreparationData] = []
  for corpus_name, step_name in corpora:
    corpus_dir = get_corpus_dir(base_dir, corpus_name)
    step_dir = get_step_dir(corpus_dir, step_name)
    if step_dir.exists():
      corpus = load_corpus(step_dir)
      corpora_data.append(corpus)
    else:
      logger.error(f"Corpus does not exist: {corpus_name} / {step_name}!")
      return

  data, selection = merge(corpora_data)

  merge_dir.mkdir(parents=True, exist_ok=False)
  save_data(merge_dir, data)

  script_dir = get_script_dir(merge_dir, script_name)
  script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(script_dir, selection)
  save_scripts(script_dir, data, selection)
  logger.info("Done.")


def app_merge_merged(base_dir: Path, merge_names: List[Tuple[str, str]], out_merge_name: str, out_script_name: str, overwrite: bool):
  logger = getLogger(__name__)
  logger.info(f"Merging multiple merged data...")

  merged_data: List[Tuple[ScriptData, Selection]] = []
  out_merge_dir = get_merge_dir(base_dir, out_merge_name)

  if out_merge_dir.exists():
    if overwrite:
      rmtree(out_merge_dir)
      logger.info("Removed existing merged data.")
    else:
      logger.info("Already exists.")
      return

  out_script_dir = get_script_dir(out_merge_dir, out_script_name)
  assert not out_script_dir.exists()

  for merge_name, in_script_name in merge_names:
    merge_dir = get_merge_dir(base_dir, merge_name)
    if not merge_dir.exists():
      logger.error("In merge dir does not exist!")
      return

    in_script_dir = get_script_dir(merge_dir, in_script_name)
    if not in_script_dir.exists():
      logger.error("In script dir does not exist!")
      return

    data = load_data(merge_dir)
    selection = load_selection(in_script_dir)
    merged_data.append((data, selection))

  res_data, res_selection = merge_merged(merged_data)

  out_merge_dir.mkdir(parents=False, exist_ok=False)
  save_data(out_merge_dir, res_data)

  out_script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(out_script_dir, res_selection)
  save_scripts(out_script_dir, res_data, res_selection)
  logger.info("Done.")


def app_select_rest(base_dir: Path, merge_name: str, in_script_name: str, out_script_name: str, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Selecting rest...")

  merge_dir = get_merge_dir(base_dir, merge_name)
  if not merge_dir.exists():
    logger.error("Merge dir does not exist!")
    return

  in_script_dir = get_script_dir(merge_dir, in_script_name)
  if not in_script_dir.exists():
    logger.error("In script dir does not exist!")
    return

  out_script_dir = get_script_dir(merge_dir, out_script_name)
  if out_script_dir.exists() and not overwrite:
    logger.error("Out script dir does already exist!")
    return

  data = load_data(merge_dir)
  selection = load_selection(in_script_dir)
  new_selection = select_rest(selection)

  if out_script_dir.exists():
    rmtree(out_script_dir)
    logger.info("Overwriting existing out script dir.")
  out_script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(out_script_dir, new_selection)
  save_scripts(out_script_dir, data, new_selection)
  logger.info("Done.")


def app_select_greedy_ngrams_epochs(base_dir: Path, merge_name: str, in_script_name: str, out_script_name: str, n_gram: int, epochs: int, overwrite: bool):
  logger = getLogger(__name__)
  logger.info(f"Selecting greedy {n_gram}-gram epochs {epochs}x...")

  merge_dir = get_merge_dir(base_dir, merge_name)
  if not merge_dir.exists():
    logger.error("Merge dir does not exist!")
    return

  in_script_dir = get_script_dir(merge_dir, in_script_name)
  if not in_script_dir.exists():
    logger.error("In script dir does not exist!")
    return

  out_script_dir = get_script_dir(merge_dir, out_script_name)
  if out_script_dir.exists() and not overwrite:
    logger.error("Out script dir does already exist!")
    return

  data = load_data(merge_dir)
  selection = load_selection(in_script_dir)
  new_selection = select_greedy_ngrams_epochs(
    data=data,
    selection=selection,
    n_gram=n_gram,
    epochs=epochs,
    ignore_symbols=None,
  )

  if out_script_dir.exists():
    rmtree(out_script_dir)
    logger.info("Overwriting existing out script dir.")
  out_script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(out_script_dir, new_selection)
  save_scripts(out_script_dir, data, new_selection)
  logger.info("Done.")


def app_select_kld_ngrams_epochs(base_dir: Path, merge_name: str, in_script_name: str, out_script_name: str, n_gram: int, epochs: int, overwrite: bool):
  logger = getLogger(__name__)
  logger.info(f"Selecting greedy {n_gram}-gram epochs {epochs}x...")

  merge_dir = get_merge_dir(base_dir, merge_name)
  if not merge_dir.exists():
    logger.error("Merge dir does not exist!")
    return

  in_script_dir = get_script_dir(merge_dir, in_script_name)
  if not in_script_dir.exists():
    logger.error("In script dir does not exist!")
    return

  out_script_dir = get_script_dir(merge_dir, out_script_name)
  if out_script_dir.exists() and not overwrite:
    logger.error("Out script dir does already exist!")
    return

  data = load_data(merge_dir)
  selection = load_selection(in_script_dir)
  new_selection = select_kld_ngrams_epochs(
    data=data,
    selection=selection,
    n_gram=n_gram,
    epochs=epochs,
    ignore_symbols=None,
  )

  if out_script_dir.exists():
    rmtree(out_script_dir)
    logger.info("Overwriting existing out script dir.")
  out_script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(out_script_dir, new_selection)
  save_scripts(out_script_dir, data, new_selection)
  logger.info("Done.")


def app_ignore(base_dir: Path, merge_name: str, in_script_name: str, out_script_name: str, ignore_symbol: Optional[str], overwrite: bool):
  logger = getLogger(__name__)
  if ignore_symbol is not None:
    logger.info(f"Ignoring utterances containing symbol \"{ignore_symbol}\"...")
  else:
    logger.info("Nothing to do.")
    return
  merge_dir = get_merge_dir(base_dir, merge_name)
  if not merge_dir.exists():
    logger.error("Merge dir does not exist!")
    return

  in_script_dir = get_script_dir(merge_dir, in_script_name)
  if not in_script_dir.exists():
    logger.error("In script dir does not exist!")
    return

  out_script_dir = get_script_dir(merge_dir, out_script_name)
  if out_script_dir.exists() and not overwrite:
    logger.error("Out script dir does already exist!")
    return

  data = load_data(merge_dir)
  selection = load_selection(in_script_dir)
  new_selection = ignore(
    data=data,
    selection=selection,
    ignore_symbol=ignore_symbol,
  )

  if out_script_dir.exists():
    rmtree(out_script_dir)
    logger.info("Overwriting existing out script dir.")
  out_script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(out_script_dir, new_selection)
  save_scripts(out_script_dir, data, new_selection)
  logger.info("Done.")


def app_log_stats(base_dir: Path, merge_name: str, script_name: str, avg_chars_per_s: int = 25):
  logger = getLogger(__name__)
  logger.info("Stats")
  merge_dir = get_merge_dir(base_dir, merge_name)
  if not merge_dir.exists():
    logger.error("Merge dir does not exist!")
    return

  in_script_dir = get_script_dir(merge_dir, script_name)
  if not in_script_dir.exists():
    logger.error("In script dir does not exist!")
    return

  data = load_data(merge_dir)
  selection = load_selection(in_script_dir)
  log_stats(
    data=data,
    selection=selection,
    avg_chars_per_s=avg_chars_per_s,
  )

  logger.info("Done.")
