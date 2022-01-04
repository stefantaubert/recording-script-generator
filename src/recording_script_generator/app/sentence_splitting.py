from pathlib import Path
from typing import Optional

from recording_script_generator.app.helper import (
    raise_error_if_directory_exists_and_not_overwrite,
    raise_error_if_directory_not_exists)
from recording_script_generator.app.io import (load_reading_passages,
                                               load_reading_passages_paths,
                                               load_selection,
                                               save_reading_passages,
                                               save_reading_passages_paths,
                                               save_representations,
                                               save_selection)
from recording_script_generator.core.sentence_splitting import main_inplace
from recording_script_generator.globals import (DEFAULT_CHUNKSIZE_FILES,
                                                DEFAULT_MAXTASKSPERCHILD,
                                                DEFAULT_N_JOBS,
                                                DEFAULT_OVERWRITE)


def app_split_sentences(working_directory: Path, custom_output_directory: Optional[Path] = None, n_jobs: int = DEFAULT_N_JOBS, maxtasksperchild: Optional[int] = DEFAULT_MAXTASKSPERCHILD, chunksize: Optional[int] = DEFAULT_CHUNKSIZE_FILES, overwrite: bool = DEFAULT_OVERWRITE):
  if raise_error_if_directory_not_exists(working_directory):
    return

  output_directory = working_directory
  if custom_output_directory is not None:
    if raise_error_if_directory_exists_and_not_overwrite(custom_output_directory, overwrite):
      return
    output_directory = custom_output_directory

  selection = load_selection(working_directory)
  reading_passages = load_reading_passages(working_directory)
  reading_passages_paths = load_reading_passages_paths(working_directory)

  representations = main_inplace(selection, reading_passages, reading_passages_paths,
                                 n_jobs, maxtasksperchild, chunksize)

  save_reading_passages(output_directory, reading_passages)
  save_selection(output_directory, selection)
  save_reading_passages_paths(output_directory, reading_passages_paths)
  save_representations(output_directory, representations)
  # TODO maybe also remove unused paths from paths
