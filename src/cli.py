import os
from argparse import ArgumentParser
from math import inf
from pathlib import Path
from typing import Callable

from text_utils import EngToIpaMode, Language

from recording_script_generator.app.main import (
    AVG_CHARS_PER_S, app_add_corpus_from_text, app_add_corpus_from_text_file,
    app_convert_to_ipa, app_generate_scripts, app_log_stats, app_merge_merged,
    app_normalize, app_remove_duplicate_utterances, app_remove_undesired_text,
    app_remove_utterances_with_acronyms,
    app_remove_utterances_with_proper_names,
    app_remove_utterances_with_too_seldom_words,
    app_remove_utterances_with_undesired_sentence_lengths,
    app_remove_utterances_with_unknown_words, app_select_all,
    app_select_greedy_ngrams_duration, app_select_greedy_ngrams_epochs,
    app_select_kld_ngrams_duration)
from recording_script_generator.core.export import SortingMode
from recording_script_generator.core.main import PreparationTarget
from recording_script_generator.utils import (parse_set, parse_tuple_list,
                                              try_parse_tuple_list)

BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=Path(base_dir))


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
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False)
  return app_add_corpus_from_text_file


def init_add_corpus_from_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False)
  return app_add_corpus_from_text


def init_log_stats_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  return app_log_stats


def init_generate_scripts_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--sorting_mode', choices=SortingMode,
                      type=SortingMode.__getitem__, required=True)
  return app_generate_scripts


def init_merge_merged_parser(parser: ArgumentParser):
  parser.add_argument('--corpora_step_names', type=str, required=True)
  parser.add_argument('--out_corpus_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=True)
  parser.add_argument('--overwrite', action='store_true')
  return _merge_merged_cli


def _merge_merged_cli(**args):
  args["corpora_step_names"] = parse_tuple_list(args["corpora_step_names"])
  app_merge_merged(**args)


def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', choices=PreparationTarget,
                      type=PreparationTarget.__getitem__, required=True)
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False)
  return app_normalize


def init_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', choices=PreparationTarget,
                      type=PreparationTarget.__getitem__, required=True)
  parser.add_argument('--mode', choices=EngToIpaMode, type=EngToIpaMode.__getitem__, required=False)
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--replace_unknown_with', type=str, default="_")
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(ignore_arcs=True, ignore_tones=False,
                      consider_ipa_annotations=False, use_cache=True)
  return app_convert_to_ipa


def init_select_all_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_select_all


def init_remove_undesired_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--undesired', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return _app_remove_undesired_text_cli


def _app_remove_undesired_text_cli(**args):
  args["undesired"] = parse_set(args["undesired"], split_symbol=" ")
  app_remove_undesired_text(**args)


IN_STEP_NAME_HELP = ""
CORPUS_NAME_HELP = ""
OUT_STEP_NAME_HELP = ""
OVERWRITE_HELP = ""


def init_remove_duplicate_utterances_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_duplicate_utterances


def init_remove_utterances_with_proper_names_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_proper_names


def init_remove_utterances_with_acronyms_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_acronyms


def init_remove_utterances_with_undesired_sentence_lengths_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
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
  parser.add_argument('--reading_speed_chars_per_s', type=int,
                      default=AVG_CHARS_PER_S)
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
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True,
                      help=IN_STEP_NAME_HELP)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=int,
                      default=AVG_CHARS_PER_S)
  parser.add_argument('--ignore_symbols', type=str, required=False)
  parser.add_argument('--boundary_min_s', type=int, default=0,
                      help="")
  parser.add_argument('--boundary_max_s', type=int, default=inf,
                      help="")
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_kld_ngrams_duration_cli


def _app_select_kld_ngrams_duration_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol=" ")
  app_select_kld_ngrams_duration(**args)


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "corpus-add", init_add_corpus_from_text_file_parser)
  _add_parser_to(subparsers, "corpus-normalize", init_normalize_parser)
  _add_parser_to(subparsers, "corpus-to-ipa", init_convert_to_ipa_parser)
  _add_parser_to(subparsers, "script-merge", init_merge_parser)
  _add_parser_to(subparsers, "script-select-rest", init_select_rest_parser)
  _add_parser_to(subparsers, "script-merge-merged", init_merge_merged_parser)
  _add_parser_to(subparsers, "script-select-greedy-ngrams-epochs",
                 init_select_greedy_ngrams_epochs_parser)
  _add_parser_to(subparsers, "script-select-kld-ngrams-epochs",
                 init_select_kld_ngrams_epochs_parser)
  _add_parser_to(subparsers, "script-ignore", init_ignore_parser)
  _add_parser_to(subparsers, "script-print-stats", init_log_stats_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()

  _process_args(received_args)
