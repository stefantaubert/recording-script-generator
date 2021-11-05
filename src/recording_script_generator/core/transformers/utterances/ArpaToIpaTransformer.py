from concurrent.futures.process import ProcessPoolExecutor
from logging import getLogger
from typing import Optional

from recording_script_generator.core.estimators.utterances.UtteranceEstimatorBase import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import (Utterance, Utterances,
                                                   clone_utterances)
from text_utils import SymbolFormat, symbols_map_arpa_to_ipa
from text_utils.types import Symbols
from tqdm import tqdm


def main(symbols: Symbols) -> Symbols:
  ipa_symbols = symbols_map_arpa_to_ipa(
    arpa_symbols=symbols,
    ignore={},
    replace_unknown=False,
    replace_unknown_with=None,
  )
  return ipa_symbols


def get_utterances_mapped_from_arpa_to_ipa(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Utterances:
  logger = getLogger(__name__)
  logger.info("Mapping ARPA to IPA...")

  result = Utterances(execute_method_on_utterances_mp(
    utterances=utterances,
    method=main,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  ))

  result.language = utterances.language
  result.symbol_format = SymbolFormat.PHONEMES_IPA

  return result


class ArpaToIpaTransformer():
  def fit(self, n_jobs: int, chunksize: int):
    self.n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    logger = getLogger(__name__)
    logger.info("Mapping ARPA to IPA...")
    with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
      res = Utterances(
        tqdm(ex.map(main, utterances.items(),
                    chunksize=self.chunksize), total=len(utterances))
      )
    result = clone_utterances(utterances)
    result.update(res)
    result.symbol_format = SymbolFormat.PHONEMES_IPA
    return result
