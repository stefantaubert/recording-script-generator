from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterances
from text_utils import change_ipa as change_ipa_method


def change_utterances_ipa_inplace(utterances: Utterances, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  method = partial(
    change_ipa_method,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
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
