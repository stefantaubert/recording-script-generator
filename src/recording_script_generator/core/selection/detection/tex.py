import re
from logging import getLogger

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances

IDS_TEX_PATTERN = re.compile(" % ([0-9]*)\n")


def detect_ids_from_tex(tex: str) -> OrderedSet[UtteranceId]:
  res = OrderedSet()
  matches = re.findall(IDS_TEX_PATTERN, tex)
  for match in matches:
    res.add(int(match))
  return res


def get_utterances_from_tex(utterances: Utterances, tex: str) -> OrderedSet[UtteranceId]:
  ids_in_tex = detect_ids_from_tex(tex)
  existent_utterance_ids = set(utterances.keys())
  valid_ids_in_tex = ids_in_tex.intersection(existent_utterance_ids)
  non_existent_utterance_ids = ids_in_tex - existent_utterance_ids
  if len(non_existent_utterance_ids) > 0:
    logger = getLogger(__name__)
    logger.warn(f"Ignored unknown ids: {', '.join(map(str, sorted(non_existent_utterance_ids)))}!")
  return valid_ids_in_tex

