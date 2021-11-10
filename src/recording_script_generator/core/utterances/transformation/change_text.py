from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterance, Utterances
from text_utils import change_symbols
from text_utils.language import Language


def main(utterance: Utterance, remove_space_around_punctuation: bool, language: Language) -> Utterance:
  assert isinstance(utterance, tuple)

  return change_symbols(
    symbols=utterance,
    remove_space_around_punctuation=remove_space_around_punctuation,
    lang=language,
  )


def change_utterances_text_inplace(utterances: Utterances, remove_space_around_punctuation: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)
  logger.info("Changing text...")
  method = partial(
    main,
    remove_space_around_punctuation=remove_space_around_punctuation,
    language=utterances.language,
  )

  result = execute_method_on_utterances_mp(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  utterances.update(result)
