from logging import getLogger
from typing import Optional

from recording_script_generator.core.multiprocessing_helper import \
    execute_method_on_utterances_mp
from recording_script_generator.core.types import Utterance, Utterances
from text_utils import SymbolFormat, symbols_map_arpa_to_ipa


def main(utterance: Utterance) -> Utterance:
  assert isinstance(utterance, tuple)
  ipa_symbols = symbols_map_arpa_to_ipa(
    arpa_symbols=utterance,
    ignore={},
    replace_unknown=False,
    replace_unknown_with=None,
  )
  return ipa_symbols


def map_utterances_from_arpa_to_ipa_inplace(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)
  logger.info("Mapping ARPA to IPA...")

  result = execute_method_on_utterances_mp(
    utterances=utterances,
    method=main,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  utterances.update(result)
  utterances.symbol_format = SymbolFormat.PHONEMES_IPA
