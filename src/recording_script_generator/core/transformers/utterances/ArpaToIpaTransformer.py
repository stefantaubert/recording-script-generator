from concurrent.futures.process import ProcessPoolExecutor
from logging import getLogger

from recording_script_generator.core.types import Utterance, Utterances, clone_utterances
from text_utils import SymbolFormat, symbols_map_arpa_to_ipa
from tqdm import tqdm


def map_to_tup_ipa(utterance: Utterance) -> Utterance:
  utterance_id, arpa_symbols = utterance
  ipa_symbols = symbols_map_arpa_to_ipa(
    arpa_symbols=arpa_symbols,
    ignore={},
    replace_unknown=False,
    replace_unknown_with=None,
  )
  return utterance_id, ipa_symbols


class ArpaToIpaTransformer():
  def fit(self, n_jobs: int, chunksize: int):
    self.n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    logger = getLogger(__name__)
    logger.info("Mapping ARPA to IPA...")
    with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
      res = Utterances(
        tqdm(ex.map(map_to_tup_ipa, utterances.items(),
                    chunksize=self.chunksize), total=len(utterances))
      )
    result = clone_utterances(utterances)
    result.update(res)
    result.symbol_format = SymbolFormat.PHONEMES_IPA
    return result
