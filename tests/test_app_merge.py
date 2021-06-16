from pathlib import Path

from recording_script_generator.app.merge import app_merge
from recording_script_generator.app.preparation import (
    add_corpus_from_text_file, app_convert_to_ipa, app_normalize)
from recording_script_generator.core.preparation import PreparationTarget
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.language import Language
from text_utils.text import EngToIpaMode


def test_app_merge(tmp_path: Path):
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
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
  )

  assert (base_dir / "scripts" / "merge1" / "data.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "rest.csv").exists()
