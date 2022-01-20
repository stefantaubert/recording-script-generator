import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

from general_utils import parse_set

from recording_script_generator.app import *
from recording_script_generator.app.exporting import (
    init_app_generate_deselected_script_parser,
    init_app_generate_selected_script_parser, init_generate_textgrid_parser)
from recording_script_generator.app.importing import init_app_add_files_parser
from recording_script_generator.app.sentence_splitting import \
    app_split_sentences
from recording_script_generator.app.transformation import app_replace
from recording_script_generator.globals import (DEFAULT_AVG_CHARS_PER_S,
                                                DEFAULT_BATCHES,
                                                DEFAULT_CHUNKSIZE_UTTERANCES,
                                                DEFAULT_MAXTASKSPERCHILD,
                                                DEFAULT_N_JOBS, DEFAULT_SEED,
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
  if set_base_dir and BASE_DIR_VAR in os.environ.keys():
    add_base_dir(parser)
  return parser


def init_log_stats_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--step_name', type=str, required=True)
  parser.add_argument('--reading_speed_chars_per_s', type=float,
                      default=DEFAULT_AVG_CHARS_PER_S)
  return app_log_stats


def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--overwrite', action='store_true')
  return app_normalize


def init_replace_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--replace_target', type=str)
  parser.add_argument('--replace_with', type=str)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--overwrite', action='store_true')
  return app_replace


def init_convert_to_arpa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_convert_to_arpa


def init_app_map_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_map_to_ipa


def init_app_app_convert_to_symbols_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_convert_to_symbols


def init_change_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ignore_stress', action='store_true')
  parser.add_argument('--break_n_thongs', action='store_true')
  parser.add_argument('--build_n_thongs', action='store_true')
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_change_ipa


def init_change_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--target', type=Target.__getitem__,
                      choices=Target, required=True, help=TARGET_HELP)
  parser.add_argument('--remove_space_around_punctuation', action='store_true')
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_change_text


def init_select_from_tex_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_select_from_tex


# def init_select_all_parser(parser: ArgumentParser):
#   parser.add_argument('--corpus_name', type=str, required=True)
#   parser.add_argument('--in_step_name', type=str, required=True)
#   parser.add_argument('--out_step_name', type=str, required=False)
#   parser.add_argument('--overwrite', action='store_true')
#   return app_select_all


# def init_deselect_all_parser(parser: ArgumentParser):
#   parser.add_argument('--corpus_name', type=str, required=True)
#   parser.add_argument('--in_step_name', type=str, required=True)
#   parser.add_argument('--out_step_name', type=str, required=False)
#   parser.add_argument('--overwrite', action='store_true')
#   return app_deselect_all


def init_remove_deselected_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return app_remove_deselected


def init_remove_undesired_text_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True)
  parser.add_argument('--in_step_name', type=str, required=True)
  parser.add_argument('--undesired', type=str, required=True)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False)
  parser.add_argument('--overwrite', action='store_true')
  return _app_remove_undesired_text_cli


def _app_remove_undesired_text_cli(**args):
  args["undesired"] = parse_set(args["undesired"], split_symbol="|")
  print("Parsed undesired: ", args["undesired"])
  app_remove_undesired_text(**args)


def init_remove_duplicate_utterances_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_duplicate_utterances


def init_remove_utterances_with_proper_names_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_proper_names


def init_remove_utterances_with_acronyms_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--min_acronym_len', type=int, default=3)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False, help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
  return app_remove_utterances_with_acronyms


def init_remove_utterances_with_undesired_sentence_lengths_parser(parser: ArgumentParser):
  parser.add_argument('--corpus_name', type=str, required=True, help=CORPUS_NAME_HELP)
  parser.add_argument('--in_step_name', type=str, required=True, help=IN_STEP_NAME_HELP)
  parser.add_argument('--min_count', type=int, required=False, help="")
  parser.add_argument('--max_count', type=int, required=False, help="")
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
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
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
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
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--batches', type=int, default=DEFAULT_BATCHES)
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return app_remove_utterances_with_too_seldom_words


def init_select_greedy_ngrams_duration_parser(parser: ArgumentParser):
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
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=None)
  parser.add_argument('--batches', type=int, default=None)
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_greedy_ngrams_duration_cli


def _app_select_greedy_ngrams_duration_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol="&")
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
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=None)
  parser.add_argument('--batches', type=int, default=None)
  parser.add_argument('--out_step_name', type=str, required=False,
                      help=OUT_STEP_NAME_HELP)
  parser.add_argument('--overwrite', action='store_true',
                      help=OVERWRITE_HELP)
  return _app_select_kld_ngrams_duration_cli


def _app_select_kld_ngrams_duration_cli(**args):
  args["ignore_symbols"] = parse_set(args["ignore_symbols"], split_symbol="&")
  app_select_kld_ngrams_duration(**args)


def init_app_split_sentences_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('--custom-output-directory', type=Path, required=False)
  parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
  parser.add_argument('--maxtasksperchild', type=int, default=DEFAULT_MAXTASKSPERCHILD)
  parser.add_argument('--chunksize', type=int, default=DEFAULT_CHUNKSIZE_UTTERANCES)
  parser.add_argument('--overwrite', action='store_true')
  return app_split_sentences


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')
  _add_parser_to(subparsers, "create", init_app_add_files_parser)
  _add_parser_to(subparsers, "split-sentences", init_app_split_sentences_parser)
  _add_parser_to(subparsers, "normalize", init_normalize_parser)
  _add_parser_to(subparsers, "replace", init_replace_parser)
  _add_parser_to(subparsers, "change-text", init_change_text_parser)
  _add_parser_to(subparsers, "eng-to-arpa", init_convert_to_arpa_parser)
  _add_parser_to(subparsers, "to-symbols", init_app_app_convert_to_symbols_parser)
  _add_parser_to(subparsers, "arpa-to-ipa", init_app_map_to_ipa_parser)
  _add_parser_to(subparsers, "change-ipa", init_change_ipa_parser)
  _add_parser_to(subparsers, "stats", init_log_stats_parser)
  _add_parser_to(subparsers, "gen-selected-script", init_app_generate_selected_script_parser)
  _add_parser_to(subparsers, "gen-deselected-script", init_app_generate_deselected_script_parser)
  _add_parser_to(subparsers, "gen-textgrid", init_generate_textgrid_parser)
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
  # _add_parser_to(subparsers, "select-all", init_select_all_parser)
  _add_parser_to(subparsers, "select-greedy-duration", init_select_greedy_ngrams_duration_parser)
  _add_parser_to(subparsers, "select-kld-duration", init_select_kld_ngrams_duration_parser)
  # _add_parser_to(subparsers, "deselect-all", init_select_all_parser)

  return result


def configure_logger() -> None:
  loglevel = logging.DEBUG  # if __debug__ else logging.INFO
  main_logger = logging.getLogger()
  main_logger.setLevel(loglevel)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  logging_formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
    '%Y/%m/%d %H:%M:%S',
  )
  console.setFormatter(logging_formatter)
  console.setLevel(loglevel)


__version__ = "0.0.1"

INVOKE_HANDLER_VAR = "invoke_handler"


def main():
  configure_logger()
  parser = _init_parser()
  received_args = parser.parse_args()
  params = vars(received_args)

  if INVOKE_HANDLER_VAR in params:
    invoke_handler: Callable = params.pop(INVOKE_HANDLER_VAR)
    invoke_handler(**params)
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
