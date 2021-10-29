import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

from general_utils import parse_set, parse_tuple_list
from text_utils import EngToIPAMode, Language
from text_utils.symbol_format import SymbolFormat

from recording_script_generator.app.main import (
    app_add_corpus_from_text, app_add_corpus_from_text_file,
    app_add_corpus_from_text_files, app_change_ipa, app_change_text,
    app_convert_to_arpa, app_deselect_all, app_generate_scripts,
    app_generate_textgrid, app_log_stats, app_merge, app_normalize,
    app_remove_deselected, app_remove_duplicate_utterances,
    app_remove_undesired_text, app_remove_utterances_with_acronyms,
    app_remove_utterances_with_proper_names,
    app_remove_utterances_with_too_seldom_words,
    app_remove_utterances_with_undesired_sentence_lengths,
    app_remove_utterances_with_unknown_words, app_select_all,
    app_select_from_tex, app_select_greedy_ngrams_duration,
    app_select_greedy_ngrams_epochs, app_select_kld_ngrams_duration)
from recording_script_generator.core.export import SortingMode
from recording_script_generator.core.main import PreparationTarget
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_SEED,
                                                DEFAULT_SPLIT_BOUNDARY_MAX_S,
                                                DEFAULT_SPLIT_BOUNDARY_MIN_S)

BASE_DIR_VAR = "base_dir"


IN_STEP_NAME_HELP = ""
CORPUS_NAME_HELP = ""
OUT_STEP_NAME_HELP = ""
OVERWRITE_HELP = ""
TARGET_HELP = ""


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = Path(os.environ[BASE_DIR_VAR])
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method: Callable[[ArgumentParser], Callable], set_base_dir: bool = True) -> ArgumentParser:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  if set_base_dir:
    add_base_dir(parser)
  return parser


def init_add_corpus_from_text_file_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text_path', type=Path, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--text_format', choices=SymbolFormat, type=SymbolFormat.__getitem__)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False)
  return app_add_corpus_from_text_file


def init_add_corpus_from_text_files_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text_dir', type=Path, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--text_format', choices=SymbolFormat, type=SymbolFormat.__getitem__)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults()
  return app_add_corpus_from_text_files


def init_add_corpus_from_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--text_format', choices=SymbolFormat, type=SymbolFormat.__getitem__)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False)
  return app_add_corpus_from_text


def init_log_stats_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=float,
                      default=DEFAULT_AVG_CHARS_PER_S)
  return app_log_stats


def init_generate_scripts_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--sorting_mode', choices=SortingMode,
                      type=SortingMode.__getitem__, required=True)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--parts_count', type=int, required=False)
  parser.add_argument('--take_per_part', type=int, required=False)
  parser.add_argument('--ignore_symbols', type=str, required=False)
  return _app_generate_scripts_cli


def _app_generate_scripts_cli(**args):
  if args["ignore_symbols"] is not None:
    args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol="&")
  app_generate_scripts(**args)


def init_merge_parser(parser: ArgumentParser):
  parser.add_argument('--corpora_step_names', type=str, required=True)
  parser.add_argument('--out_corpus_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=True)
  parser.add_argument('--overwrite', action='store_true')
  return _merge_cli


def _merge_cli(**args):
  args["corpora_step_names"] = parse_tuple_list(args["corpora_step_names"])
  app_merge(**args)


def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_normalize


def init_convert_to_arpa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  #parser.add_argument('--mode', choices=EngToIPAMode, type=EngToIPAMode.__getitem__, required=False)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_convert_to_arpa


def init_change_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ignore_stress', action='store_true')
  parser.add_argument('--break_n_thongs', action='store_true')
  parser.add_argument('--build_n_thongs', action='store_true')
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_change_ipa


def init_change_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--remove_space_around_punctuation', action='store_true')
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_change_text


def init_select_from_tex_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_select_from_tex


def init_select_all_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_select_all


def init_deselect_all_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_deselect_all


def init_remove_deselected_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_remove_deselected


def init_remove_undesired_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--undesired', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return _app_remove_undesired_text_cli


def _app_remove_undesired_text_cli(**args):
  args["undesired"] = parse_set(args["undesired"], split_symbol=" ")
  app_remove_undesired_text(**args)


def init_remove_duplicate_utterances_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_duplicate_utterances


def init_remove_utterances_with_proper_names_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_proper_names


def init_remove_utterances_with_acronyms_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_acronyms


def init_remove_utterances_with_undesired_sentence_lengths_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--min_word_count', type=int, required=False, help="")
  parser.add_argument('--max_word_count', type=int, required=False, help="")
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_undesired_sentence_lengths


def init_remove_utterances_with_unknown_words_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name',
                      type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name',
                      type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--max_unknown_word_count',
                      type=int, required=False, help="")
  parser.add_argument('--out_step_name',
                      type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite',
                      action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_unknown_words


def init_remove_utterances_with_too_seldom_words_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True,
                      help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True,
                      help=IN_STEP_NAME_HELP)
  parser.add_argument('--target', type=PreparationTarget.__getitem__,
                      choices=PreparationTarget, required=True, help=TARGET_HELP)
  parser.add_argument('--min_occurrence_count', type=int, required=False,
                      help="")
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return app_remove_utterances_with_too_seldom_words


def init_select_greedy_ngrams_epochs_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True,
                      help=IN_STEP_NAME_HELP)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--epochs', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str, required=False)
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_greedy_ngrams_epochs_cli


def _app_select_greedy_ngrams_epochs_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol=" ")
  app_select_greedy_ngrams_epochs(**args)


def init_select_greedy_ngrams_duration_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True,
                      help=IN_STEP_NAME_HELP)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=float,
                      default=DEFAULT_AVG_CHARS_PER_S)
  parser.add_argument('--ignore_symbols', type=str, required=False)
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_greedy_ngrams_duration_cli


def _app_select_greedy_ngrams_duration_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol=" ")
  app_select_greedy_ngrams_duration(**args)


def init_select_kld_ngrams_duration_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True,
                      help=IN_STEP_NAME_HELP)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=float,
                      default=DEFAULT_AVG_CHARS_PER_S)
  parser.add_argument('--ignore_symbols', type=str, required=False)
  parser.add_argument('--boundary_min_s', type=float,
                      default=DEFAULT_SPLIT_BOUNDARY_MIN_S, help="")
  parser.add_argument('--boundary_max_s', type=float,
                      default=DEFAULT_SPLIT_BOUNDARY_MAX_S, help="")
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_kld_ngrams_duration_cli


def _app_select_kld_ngrams_duration_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol="&")
  app_select_kld_ngrams_duration(**args)


def init_generate_textgrid_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=float,
                      default=DEFAULT_AVG_CHARS_PER_S)
  return app_generate_textgrid


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')
  _add_parser_to(subparsers, "add-file", init_add_corpus_from_text_file_parser)
  _add_parser_to(subparsers, "add-files", init_add_corpus_from_text_files_parser)
  _add_parser_to(subparsers, "add-text", init_add_corpus_from_text_parser)
  _add_parser_to(subparsers, "normalize", init_normalize_parser)
  _add_parser_to(subparsers, "change-text", init_change_text_parser)
  _add_parser_to(subparsers, "to-arpa", init_convert_to_arpa_parser)
  _add_parser_to(subparsers, "change-ipa", init_change_ipa_parser)
  _add_parser_to(subparsers, "stats", init_log_stats_parser)
  _add_parser_to(subparsers, "gen-scripts", init_generate_scripts_parser)
  _add_parser_to(subparsers, "gen-textgrid", init_generate_textgrid_parser)
  _add_parser_to(subparsers, "merge", init_merge_parser)
  _add_parser_to(subparsers, "remove-deselected", init_remove_deselected_parser)
  _add_parser_to(subparsers, "remove-text", init_remove_undesired_text_parser)
  _add_parser_to(subparsers, "remove-duplicates", init_remove_duplicate_utterances_parser)
  _add_parser_to(subparsers, "remove-proper-names", init_remove_utterances_with_proper_names_parser)
  _add_parser_to(subparsers, "remove-acronyms", init_remove_utterances_with_acronyms_parser)
  _add_parser_to(subparsers, "remove-word-counts",
                 init_remove_utterances_with_undesired_sentence_lengths_parser)
  _add_parser_to(subparsers, "remove-unknown-words",
                 init_remove_utterances_with_unknown_words_parser)
  _add_parser_to(subparsers, "remove-rare-words",
                 init_remove_utterances_with_too_seldom_words_parser)
  _add_parser_to(subparsers, "select-from-tex", init_select_from_tex_parser)
  _add_parser_to(subparsers, "select-all", init_select_all_parser)
  _add_parser_to(subparsers, "select-greedy-epochs", init_select_greedy_ngrams_epochs_parser)
  _add_parser_to(subparsers, "select-greedy-duration", init_select_greedy_ngrams_duration_parser)
  _add_parser_to(subparsers, "select-kld-duration", init_select_kld_ngrams_duration_parser)
  _add_parser_to(subparsers, "deselect-all", init_select_all_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()

  _process_args(received_args)
