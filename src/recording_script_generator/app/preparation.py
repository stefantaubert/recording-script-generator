from logging import getLogger
from pathlib import Path
from typing import Optional

from recording_script_generator.core.preparation import (PreparationTarget,
                                                         add_corpus_from_text,
                                                         convert_to_ipa,
                                                         normalize)
from recording_script_generator.utils import (get_subdir, load_obj, read_lines,
                                              save_json, save_obj)
from text_utils import Language
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import EngToIpaMode

DATA_FILE = "data.pkl"


def get_corpus_dir(base_dir: str, corpus_name: str, create: bool = False):
  return get_subdir(base_dir, corpus_name, create)


def add_corpus_from_text_file(base_dir: str, corpus_name: str, step_name: str, text_path: Path, lang: Language, ipa_settings: Optional[IPAExtractionSettings]) -> None:
  logger = getLogger(__name__)
  corpus_dir = Path(base_dir) / corpus_name
  step_dir = corpus_dir / step_name
  if corpus_dir.exists():
    logger.info("Already exists.")
    return
  if step_dir.exists():
    logger.info("Already exists.")
    return
  if not text_path.exists():
    logger.error("File not exists.")
    return
  lines = read_lines(text_path)

  res = add_corpus_from_text(
    utterances=lines,
    lang=lang,
    ipa_settings=ipa_settings,
  )

  save_obj(
    path=step_dir / DATA_FILE,
    obj=res,
  )


def app_normalize(base_dir: str, corpus_name: str, in_step_name: str, out_step_name: str, ipa_settings: Optional[IPAExtractionSettings], target: PreparationTarget):
  logger = getLogger(__name__)
  corpus_dir = Path(base_dir) / corpus_name
  in_step_dir = corpus_dir / in_step_name
  out_step_dir = corpus_dir / out_step_name
  if not corpus_dir.exists():
    logger.info("Corpus dir does not exists.")
    return
  if not in_step_dir.exists():
    logger.info("In dir not exists.")
    return
  if out_step_dir.exists():
    logger.info("Out dir exists.")
    return
  in_data_path = in_step_dir / DATA_FILE
  assert in_data_path.exists()
  obj = load_obj(in_data_path)

  res = normalize(
    data=obj,
    target=target,
    ipa_settings=ipa_settings,
  )

  save_obj(
    path=out_step_dir / DATA_FILE,
    obj=res,
  )
  logger.info("Done.")


def app_convert_to_ipa(base_dir: str, corpus_name: str, in_step_name: str, out_step_name: str, ipa_settings: Optional[IPAExtractionSettings], target: PreparationTarget, mode: Optional[EngToIpaMode], replace_unknown_with: Optional[str] = "_", use_cache: Optional[bool] = True):
  logger = getLogger(__name__)
  corpus_dir = Path(base_dir) / corpus_name
  in_step_dir = corpus_dir / in_step_name
  out_step_dir = corpus_dir / out_step_name
  if not corpus_dir.exists():
    logger.info("Corpus dir does not exists.")
    return
  if not in_step_dir.exists():
    logger.info("In dir not exists.")
    return
  if out_step_dir.exists():
    logger.info("Out dir exists.")
    return
  in_data_path = in_step_dir / DATA_FILE
  assert in_data_path.exists()
  obj = load_obj(in_data_path)

  res = convert_to_ipa(
    data=obj,
    target=target,
    ipa_settings=ipa_settings,
    mode=mode,
    replace_unknown_with=replace_unknown_with,
    use_cache=use_cache,
  )

  save_obj(
    path=out_step_dir / DATA_FILE,
    obj=res,
  )
  logger.info("Done.")

