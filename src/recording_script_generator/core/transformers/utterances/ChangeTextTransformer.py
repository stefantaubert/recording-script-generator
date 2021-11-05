from functools import partial
from logging import getLogger
from typing import Optional

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterances, clone_utterances
from text_utils import change_symbols
from tqdm import tqdm


def get_utterances_with_changed_text(utterances: Utterances, remove_space_around_punctuation: bool, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Utterances:
  logger = getLogger(__name__)
  logger.info("Changing text...")
  method = partial(
    change_symbols,
    remove_space_around_punctuation=remove_space_around_punctuation,
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


class ChangeTextTransformer():
  def fit(self, remove_space_around_punctuation: bool, n_jobs: int, chunksize: int):
    self.remove_space_around_punctuation = remove_space_around_punctuation
    self.n_jobs = n_jobs = n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    res = clone_utterances(utterances)

    for utterance_id, symbols in tqdm(utterances.items()):
      new_symbols = change_symbols(
        symbols=symbols,
        remove_space_around_punctuation=self.remove_space_around_punctuation,
        lang=utterances.language,
      )

      res[utterance_id] = new_symbols

    return res
