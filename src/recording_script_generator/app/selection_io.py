
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

from ordered_set import OrderedSet
from recording_script_generator.app.helper import add_overwrite_argument
from recording_script_generator.app.io import (load_representations,
                                               load_selection, save_selection)
from text_selection_app.api import (Dataset, create, get_selection,
                                    get_uniform_weights)


def init_export_for_selection_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('selection_directory', type=Path, metavar="selection-directory",
                      help="directory to write the prepared files for selection")
  add_overwrite_argument(parser)
  return export_for_selection


def export_for_selection(working_directory: Path, selection_directory: Path, overwrite: bool) -> None:
  logger = getLogger(__name__)

  if not working_directory.exists():
    logger.error("Corpus does not exist!")
    return

  representations = load_representations(working_directory)
  selection = load_selection(working_directory)

  keys = OrderedSet(representations.entries.keys())
  dataset = Dataset(keys)
  dataset.subsets["Available"] = keys - selection
  dataset.subsets["Ignored"] = OrderedSet()
  dataset.subsets["Selected"] = selection

  weights = {
    "weights": get_uniform_weights(keys)
  }

  try:
    create(selection_directory, dataset, weights, representations.entries, overwrite)
  except ValueError as error:
    logger.error(error)
    return


def init_import_from_selection_parser(parser: ArgumentParser):
  parser.add_argument('working_directory', metavar="working-directory", type=Path)
  parser.add_argument('selection_directory', type=Path, metavar="selection-directory",
                      help="directory to write the prepared files for selection")
  return import_from_selection


def import_from_selection(working_directory: Path, selection_directory: Path) -> None:
  logger = getLogger(__name__)

  if not working_directory.exists():
    logger.error("Corpus does not exist!")
    return

  try:
    new_selection = get_selection(selection_directory, "Selected")
  except ValueError as error:
    logger.error(error)
    return

  old_selection = load_selection(selection_directory)
  selected_not_only_from_selection = not new_selection.issubset(old_selection)
  if selected_not_only_from_selection:
    representations = load_representations(working_directory)
    available_keys = representations.entries.keys()

    if not new_selection.issubset(available_keys):
      logger.error("Selection contains indices that do not exist!")
      return

  save_selection(working_directory, new_selection)
