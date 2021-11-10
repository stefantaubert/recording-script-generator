from logging import getLogger
from typing import Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import Utterance, UtteranceId, Utterances
from tqdm import tqdm


def get_duplicate_utterances(utterances: Utterances) -> Set[UtteranceId]:
  logger = getLogger(__name__)
  logger.info("Detecting duplicate utterances...")
  result = OrderedSet()
  already_exist: Set[Utterance] = set()
  for utterance_id, utterance in tqdm(utterances.items()):
    if utterance in already_exist:
      result.add(utterance_id)
    else:
      already_exist.add(utterance)
  return result
