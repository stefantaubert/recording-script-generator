from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Optional

from recording_script_generator.core.preparation import (PreparationData,
                                                         PreparationTarget,
                                                         add_corpus_from_text,
                                                         convert_to_ipa,
                                                         normalize)
from recording_script_generator.utils import (get_subdir, load_obj, read_lines,
                                              save_json, save_obj)
from text_utils import Language
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import EngToIpaMode

DATA_FILE = "data.pkl"
CORPORA_DIR_NAME = "corpora"


def get_corpus_dir(base_dir: Path, corpus_name: str) -> Path:
  return base_dir / CORPORA_DIR_NAME / corpus_name


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


def add_corpus_from_text_file(base_dir: Path, corpus_name: str, step_name: str, text_path: Path, lang: Language, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], replace_unknown_ipa_by: Optional[str], overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Adding corpus...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  if not text_path.exists():
    logger.error("File not exists.")
    return
  if corpus_dir.exists():
    if overwrite:
      rmtree(corpus_dir)
      logger.info("Removed existing corpus.")
    else:
      logger.info("Already exists.")
      return
  lines = read_lines(text_path)

  res = add_corpus_from_text(
    utterances=lines,
    lang=lang,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
  )

  corpus_dir.mkdir(parents=True, exist_ok=False)
  step_dir = get_step_dir(corpus_dir, step_name)
  step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(step_dir, res)


def app_normalize(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: str, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], replace_unknown_ipa_by: Optional[str], target: PreparationTarget, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Normalizing...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
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
  in_data_path = in_step_dir / DATA_FILE
  assert in_data_path.exists()
  obj = load_obj(in_data_path)

  res = normalize(
    data=obj,
    target=target,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
  )

  if out_step_dir.exists():
    rmtree(out_step_dir)
    logger.info("Overwriting existing out dir.")
  out_step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(out_step_dir, res)
  logger.info("Done.")


def app_convert_to_ipa(base_dir: Path, corpus_name: str, in_step_name: str, out_step_name: str, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], replace_unknown_ipa_by: Optional[str], target: PreparationTarget, mode: Optional[EngToIpaMode], overwrite: bool, replace_unknown_with: Optional[str] = "_", consider_ipa_annotations: bool = False, use_cache: Optional[bool] = True):
  logger = getLogger(__name__)
  logger.info("Converting to IPA...")
  corpus_dir = get_corpus_dir(base_dir, corpus_name)
  in_step_dir = get_step_dir(corpus_dir, in_step_name)
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
  in_data_path = in_step_dir / DATA_FILE
  assert in_data_path.exists()
  obj = load_obj(in_data_path)

  res = convert_to_ipa(
    data=obj,
    target=target,
    ipa_settings=IPAExtractionSettings(ignore_tones, ignore_arcs, replace_unknown_ipa_by),
    mode=mode,
    replace_unknown_with=replace_unknown_with,
    consider_ipa_annotations=consider_ipa_annotations,
    use_cache=use_cache,
  )

  if out_step_dir.exists():
    rmtree(out_step_dir)
    logger.info("Overwriting existing out dir.")
  out_step_dir.mkdir(parents=False, exist_ok=False)
  save_corpus(out_step_dir, res)
  logger.info("Done.")
