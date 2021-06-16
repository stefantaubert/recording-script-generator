from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple

from recording_script_generator.app.preparation import (get_corpus_dir,
                                                        get_step_dir,
                                                        load_corpus)
from recording_script_generator.core.merge import (ScriptData, Selection,
                                                   get_reading_scripts, merge)
from recording_script_generator.core.preparation import PreparationData
from recording_script_generator.utils import save_obj

SCRIPTS_DIR_NAME = "scripts"
DATA_FILENAME = "data.pkl"
SELECTION_FILENAME = "selection.pkl"
SELECTED_FILENAME = "selected.csv"
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


def save_selection(script_dir: Path, selection: Selection) -> None:
  path = script_dir / SELECTION_FILENAME
  save_obj(path, selection)


def save_scripts(script_dir: Path, data: ScriptData, selection: Selection) -> None:
  selected_df, ignored_df, rest_df = get_reading_scripts(data, selection)
  selected_df.to_csv(script_dir / SELECTED_FILENAME, sep=SEP, header=True)
  ignored_df.to_csv(script_dir / IGNORED_FILENAME, sep=SEP, header=True)
  rest_df.to_csv(script_dir / REST_FILENAME, sep=SEP, header=True)


def app_merge(base_dir: Path, merge_name: str, script_name: str, corpora: List[Tuple[str, str]]) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merge_dir(base_dir, merge_name)
  if merge_dir.exists():
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
      logger.error(f"Corpus does not exist: {corpus_name} / {step_dir}!")
      return

  data, selection = merge(corpora_data)

  merge_dir.mkdir(parents=True, exist_ok=False)
  save_data(merge_dir, data)

  script_dir = get_script_dir(merge_dir, script_name)
  script_dir.mkdir(parents=False, exist_ok=False)
  save_selection(script_dir, selection)
  save_scripts(script_dir, data, selection)
  logger.info("Done")
