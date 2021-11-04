from logging import getLogger
from typing import Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from text_utils.types import Symbols
from tqdm import tqdm


def contains_undesired_text(sentence: str, undesired: Set[str], ignore_case: bool) -> bool:
  for x in undesired:
    if ignore_case:
      if x.lower() in sentence.lower():
        return True
    else:
      if x in sentence:
        return True

  return False


def main(symbols: Symbols, undesired: Set[str]) -> bool:
  symbols = ''.join(symbols)
  result = contains_undesired_text(symbols, undesired=undesired, ignore_case=True)
  return result


class DuplicateEstimator():
  def fit(self):
    pass

  def estimate(self, utterances: Utterances) -> Set[UtteranceId]:
    logger = getLogger(__name__)
    logger.info("Detecting duplicate utterances...")
    result = OrderedSet()
    already_exist: Set[Symbols] = set()
    for utterance_id, utterance_symbols in tqdm(utterances.items()):
      if utterance_symbols in already_exist:
        result.add(utterance_id)
      else:
        already_exist.add(utterance_symbols)
    return result
