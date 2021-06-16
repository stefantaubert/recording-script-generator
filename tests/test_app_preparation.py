from pathlib import Path

from recording_script_generator.app.preparation import (
    add_corpus_from_text_file, app_convert_to_ipa, app_normalize)
from recording_script_generator.core.preparation import PreparationTarget
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.language import Language
from text_utils.text import EngToIpaMode


def test_add_corpus_from_text_file(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    ipa_settings=None,
    overwrite=False,
  )

  assert (base_dir / "corpora" / "corpus1" / "step1" / "data.pkl").exists()


def test_app_normalize(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line 1\nline 2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    ipa_settings=None,
    overwrite=False,
  )

  app_normalize(
    base_dir=base_dir,
    corpus_name="corpus1",
    in_step_name="step1",
    out_step_name="step2",
    target=PreparationTarget.BOTH,
    ipa_settings=None,
  )

  assert (base_dir / "corpora" / "corpus1" / "step2" / "data.pkl").exists()


def test_app_convert_to_ipa(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line 1\nline 2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    ipa_settings=None,
    overwrite=False,
  )

  app_convert_to_ipa(
    base_dir=base_dir,
    corpus_name="corpus1",
    in_step_name="step1",
    out_step_name="step2",
    target=PreparationTarget.BOTH,
    ipa_settings=IPAExtractionSettings(True, True, "_"),
    mode=EngToIpaMode.BOTH,
  )

  assert (base_dir / "corpora" / "corpus1" / "step2" / "data.pkl").exists()
