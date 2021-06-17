import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

from text_utils import EngToIpaMode, Language

from recording_script_generator.app.merge import (
    app_ignore, app_log_stats, app_merge, app_merge_merged,
    app_select_greedy_ngrams_epochs)
from recording_script_generator.app.preparation import (
    add_corpus_from_text_file, app_convert_to_ipa, app_normalize)
from recording_script_generator.app.text_extraction import app_ljs_to_text
from recording_script_generator.core.merge import select_rest
from recording_script_generator.core.preparation import PreparationTarget
from recording_script_generator.utils import parse_tuple_list

BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=Path(base_dir))


def _add_parser_to(subparsers, name: str, init_method: Callable[[ArgumentParser], Callable], set_base_dir: bool = True) -> ArgumentParser:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def init_ljs_to_text_file_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text_path', type=Path, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  return app_ljs_to_text


# region corpus


def init_add_corpus_from_text_file_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--text_path', type=Path, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.set_defaults(overwrite=True, ignore_arcs=True)
  return add_corpus_from_text_file


def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--target', choices=PreparationTarget, type=PreparationTarget.__getitem__)
  parser.set_defaults(overwrite=True, ignore_arcs=True)
  return app_normalize


def init_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--replace_unknown_ipa_by', type=str, default="_")
  parser.add_argument('--target', choices=PreparationTarget, type=PreparationTarget.__getitem__)
  parser.add_argument('--mode', choices=EngToIpaMode, type=EngToIpaMode.__getitem__)
  parser.set_defaults(overwrite=True, ignore_arcs=True, use_cache=True)
  return app_convert_to_ipa
# endregion

# region script


def init_merge_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--script_name', type=str, required=True)
  parser.add_argument('--corpora', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return _merge_cli


def _merge_cli(**args):
  args["corpora"] = parse_tuple_list(args["corpora"])
  app_merge(**args)


def init_select_rest_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_script_name', type=str, required=True)
  parser.add_argument('--out_script_name', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return select_rest


def init_select_greedy_ngrams_epochs_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_script_name', type=str, required=True)
  parser.add_argument('--out_script_name', type=str, required=True)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--epochs', type=int, required=True)
  parser.set_defaults(overwrite=True)
  return app_select_greedy_ngrams_epochs


def init_ignore_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--in_script_name', type=str, required=True)
  parser.add_argument('--out_script_name', type=str, required=True)
  parser.add_argument('--ignore_symbol', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return app_ignore


def init_log_stats_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--script_name', type=str, required=True)
  parser.add_argument('--avg_chars_per_s', type=int, default=25)
  parser.set_defaults(overwrite=True)
  return app_log_stats


def init_merge_merged_parser(parser: ArgumentParser):
  parser.add_argument('--merge_names', type=str, required=True)
  parser.add_argument('--out_merge_name', type=str, required=True)
  parser.add_argument('--out_script_name', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return _merge_merged_cli


def _merge_merged_cli(**args):
  args["merge_names"] = parse_tuple_list(args["merge_names"])
  app_merge_merged(**args)
# endregion


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "extract-text-ljs", init_ljs_to_text_file_parser, set_base_dir=False)
  _add_parser_to(subparsers, "script-merge", init_merge_parser)
  _add_parser_to(subparsers, "script-select-rest", init_select_rest_parser)
  _add_parser_to(subparsers, "script-merge-merged", init_merge_merged_parser)
  _add_parser_to(subparsers, "script-select-greedy-ngrams-epochs",
                 init_select_greedy_ngrams_epochs_parser)
  _add_parser_to(subparsers, "script-ignore", init_ignore_parser)
  _add_parser_to(subparsers, "script-print-stats", init_log_stats_parser)
  _add_parser_to(subparsers, "corpus-add", init_add_corpus_from_text_file_parser)
  _add_parser_to(subparsers, "corpus-normalize", init_normalize_parser)
  _add_parser_to(subparsers, "corpus-to-ipa", init_convert_to_ipa_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()

  _process_args(received_args)
