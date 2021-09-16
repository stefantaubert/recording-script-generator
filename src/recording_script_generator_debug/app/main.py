from math import inf
from pathlib import Path

from recording_script_generator.app.main import (
    app_add_corpus_from_text, app_convert_to_ipa, app_generate_scripts,
    app_generate_textgrid, app_log_stats, app_merge, app_normalize,
    app_remove_duplicate_utterances, app_remove_undesired_text,
    app_remove_utterances_with_acronyms,
    app_remove_utterances_with_proper_names,
    app_remove_utterances_with_too_seldom_words,
    app_remove_utterances_with_undesired_sentence_lengths,
    app_remove_utterances_with_unknown_words, app_select_all,
    app_select_greedy_ngrams_duration, app_select_greedy_ngrams_epochs,
    app_select_kld_ngrams_duration)
from recording_script_generator.core.export import SortingMode
from recording_script_generator.core.main import PreparationTarget
from recording_script_generator.utils import read_text
from text_utils import EngToIPAMode, Language
from text_utils.symbol_format import SymbolFormat

UNDESIRED = {"/", "\\", "-", ":", "@", ";", "*", "%", "\"",
             "(", ")", "[", "]", "{", "}", "quote", "oswald"}


def prepare():
  base_dir = Path("out")
  corpus_name = "debug"
  step_name = "initial"

  text = ""
  text += read_text(Path("corpora/darwin.txt"))
  text += read_text(Path("corpora/solomon.txt"))
  text += read_text(Path("corpora/ljs.txt"))
  text += read_text(Path("corpora/thoughts.txt"))
  text += read_text(Path("corpora/pg27682.txt"))
  text += read_text(Path("corpora/pg32500.txt"))
  text += read_text(Path("corpora/pg38180.txt"))
  text += read_text(Path("corpora/pg39455.txt"))
  text += read_text(Path("corpora/pg42197.txt"))

  app_add_corpus_from_text(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name=step_name,
    text=text,
    text_format=SymbolFormat.GRAPHEMES,
    lang=Language.ENG,
    overwrite=True,
  )

  app_remove_utterances_with_acronyms(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_normalize(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_utterances_with_too_seldom_words(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    min_occurrence_count=2,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_utterances_with_proper_names(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_undesired_text(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    undesired=UNDESIRED,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_duplicate_utterances(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_utterances_with_undesired_sentence_lengths(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    min_word_count=3,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_utterances_with_unknown_words(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    max_unknown_word_count=0,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_convert_to_ipa(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    mode=EngToIPAMode.EPITRAN,
    target=PreparationTarget.REPRESENTATIONS,
  )

  app_remove_undesired_text(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    undesired={"'"},
    target=PreparationTarget.REPRESENTATIONS,
  )


def main():
  # prepare()

  base_dir = Path("out")
  corpus_name = "debug"
  step_name = "initial"

  app_select_kld_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="kld_boundaries",
    n_gram=1,
    minutes=3 * 0.1,
    ignore_symbols={" ", 'ˈ'},
    boundary_min_s=0,
    boundary_max_s=3,
    overwrite=True,
  )

  text = read_text(Path("corpora/darwin.txt"))

  app_add_corpus_from_text(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name=step_name,
    text=text,
    text_format=SymbolFormat.GRAPHEMES,
    lang=Language.ENG,
    overwrite=True,
  )

  app_select_all(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    overwrite=True,
  )

  app_generate_scripts(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name=step_name,
    sorting_mode=SortingMode.BY_INDEX,
  )

  app_generate_textgrid(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name=step_name,
  )

  app_log_stats(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name=step_name,
  )

  app_select_kld_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="kld_boundaries",
    n_gram=1,
    minutes=30 * 0.1,
    ignore_symbols={" ", 'ˈ'},
    boundary_min_s=0,
    boundary_max_s=3,
    overwrite=True,
  )

  app_select_kld_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name="kld_boundaries",
    out_step_name="kld_boundaries",
    n_gram=1,
    minutes=30 * 0.4,
    ignore_symbols={" ", 'ˈ'},
    boundary_min_s=3,
    boundary_max_s=5,
    overwrite=True,
  )

  app_select_kld_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name="kld_boundaries",
    out_step_name="kld_boundaries",
    n_gram=1,
    minutes=30 * 0.5,
    ignore_symbols={" ", 'ˈ'},
    boundary_min_s=5,
    boundary_max_s=inf,
    overwrite=True,
  )

  app_generate_scripts(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name="kld_boundaries",
    sorting_mode=SortingMode.RANDOM,
  )

  return

  app_select_greedy_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="greedy_custom_selection",
    n_gram=1,
    minutes=3,
    ignore_symbols={" ", 'ˈ', "?", "!"},
  )

  return

  log_stats(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name="initial",
  )

  app_select_greedy_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="greedy_min",
    n_gram=1,
    minutes=30,
    ignore_symbols={" ", 'ˈ'},
  )

  app_select_kld_ngrams_duration(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="kld_min",
    n_gram=1,
    minutes=30,
    ignore_symbols={" ", 'ˈ'},
  )

  return

  log_stats(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name="greedy",
  )

  log_stats(
    base_dir=base_dir,
    corpus_name=corpus_name,
    step_name="kld",
  )

  app_select_greedy_ngrams_epochs(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="greedy",
    epochs=80,
    n_gram=1,
  )

  app_select_kld_ngrams_iterations(
    base_dir=base_dir,
    corpus_name=corpus_name,
    in_step_name=step_name,
    out_step_name="kld",
    iterations=900,
    n_gram=1,
    ignore_symbols={" ", 'ˈ'},
  )

  # app_select_all(
  #   base_dir=base_dir,
  #   corpus_name=corpus_name,
  #   in_step_name=step_name,
  # )

  # app_merge_merged(
  #   base_dir=base_dir,
  #   corpora_step_names=[(corpus_name, step_name), (corpus_name, step_name)],
  #   out_corpus_name=corpus_name + "_merged",
  #   out_step_name=step_name,
  #   overwrite=True,
  # )


if __name__ == "__main__":
  main()
