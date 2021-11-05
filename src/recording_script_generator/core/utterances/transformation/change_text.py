from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterances, clone_utterances
from text_utils import change_symbols
from tqdm import tqdm


def change_utterances_text_inplace(utterances: Utterances, remove_space_around_punctuation: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> None:
  logger = getLogger(__name__)
  logger.info("Changing text...")
  method = partial(
    change_symbols,
    remove_space_around_punctuation=remove_space_around_punctuation,
    language=utterances.language,
  )

  result = execute_method_on_utterances_mp(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  )

  utterances.update(result)
