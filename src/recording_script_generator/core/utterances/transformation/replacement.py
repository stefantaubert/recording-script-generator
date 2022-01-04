from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterance, Utterances
from text_utils.types import Symbol


def replace_func(utterance: Utterance, replace_target: Symbol, replace_with: Symbol) -> Utterance:
  assert isinstance(utterance, str)
  result = utterance.replace(replace_target, replace_with)
  return result


def replace_utterance_inplace(utterances: Utterances, replace_target: Symbol, replace_with: Symbol, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)
  logger.info(f"Replacing \"{str(replace_target)}\" with \"{str(replace_with)}\"...")
  method = partial(
    replace_func,
    replace_target=replace_target,
    replace_with=replace_with,
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
