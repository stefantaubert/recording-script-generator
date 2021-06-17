from pathlib import Path

from recording_script_generator.app.merge import (
    app_ignore, app_log_stats, app_merge, app_merge_merged,
    app_select_greedy_ngrams_epochs, app_select_rest)
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
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  assert (base_dir / "scripts" / "merge1" / "data.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "selected.txt").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "selected.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "ignored.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script1" / "rest.csv").exists()


def test_app_merge_merged(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge2",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_merge_merged(
    base_dir=base_dir,
    merge_names=[("merge1", "script1"), ("merge2", "script1")],
    out_merge_name="merge3",
    out_script_name="script1",
    overwrite=False,
  )

  assert (base_dir / "scripts" / "merge3" / "data.pkl").exists()
  assert (base_dir / "scripts" / "merge3" / "script1" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge3" / "script1" / "selected.txt").exists()
  assert (base_dir / "scripts" / "merge3" / "script1" / "selected.csv").exists()
  assert (base_dir / "scripts" / "merge3" / "script1" / "ignored.csv").exists()
  assert (base_dir / "scripts" / "merge3" / "script1" / "rest.csv").exists()


def test_app_select_rest(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_select_rest(
    base_dir=base_dir,
    merge_name="merge1",
    in_script_name="script1",
    out_script_name="script2",
    overwrite=False,
  )

  assert (base_dir / "scripts" / "merge1" / "script2" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.txt").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "ignored.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "rest.csv").exists()


def test_app_ignore(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_ignore(
    base_dir=base_dir,
    merge_name="merge1",
    in_script_name="script1",
    out_script_name="script2",
    ignore_symbol="2",
    overwrite=False,
  )

  assert (base_dir / "scripts" / "merge1" / "script2" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.txt").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "ignored.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "rest.csv").exists()


def test_app_log_stats(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_log_stats(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
  )

  assert True


def test_app_select_greedy_ngrams_epochs(tmp_path: Path):
  base_dir = tmp_path / "base_dir"
  text_path = tmp_path / "input.txt"
  text_path.write_text("line1\nlin1e\nline2\n")
  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="corpus1",
    step_name="step1",
    text_path=text_path,
    lang=Language.ENG,
    replace_unknown_ipa_by=None,
    ignore_arcs=None,
    ignore_tones=None,
    overwrite=False,
  )

  app_merge(
    base_dir=base_dir,
    merge_name="merge1",
    script_name="script1",
    corpora=[("corpus1", "step1")],
    overwrite=False,
  )

  app_select_greedy_ngrams_epochs(
    base_dir=base_dir,
    merge_name="merge1",
    in_script_name="script1",
    out_script_name="script2",
    n_gram=1,
    epochs=1,
    overwrite=False,
  )

  assert (base_dir / "scripts" / "merge1" / "script2" / "selection.pkl").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.txt").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "selected.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "ignored.csv").exists()
  assert (base_dir / "scripts" / "merge1" / "script2" / "rest.csv").exists()
