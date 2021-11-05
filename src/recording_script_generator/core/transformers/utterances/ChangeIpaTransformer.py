from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterances, clone_utterances
from text_utils import change_ipa as change_ipa_method
from tqdm import tqdm


def get_utterances_with_changed_ipa(utterances: Utterances, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Utterances:
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

  result = Utterances(execute_method_on_utterances_mp(
    utterances=utterances,
    method=method,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  ))

  result.language = utterances.language
  result.symbol_format = utterances.symbol_format

  return result


class ChangeIpaTransformer():
  def fit(self, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, chunksize: int):
    self.ignore_tones = ignore_tones
    self.ignore_arcs = ignore_arcs
    self.ignore_stress = ignore_stress
    self.break_n_thongs = break_n_thongs
    self.build_n_thongs = build_n_thongs
    self.n_jobs = n_jobs = n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    res = clone_utterances(utterances)

    for utterance_id, symbols in tqdm(utterances.items()):
      new_symbols = change_ipa_method(
        symbols=symbols,
        ignore_tones=self.ignore_tones,
        ignore_arcs=self.ignore_arcs,
        ignore_stress=self.ignore_stress,
        break_n_thongs=self.break_n_thongs,
        build_n_thongs=self.build_n_thongs,
        language=utterances.language,
      )

      res[utterance_id] = new_symbols

    return res
