import re

from ordered_set import OrderedSet

IDS_TEX_PATTERN = re.compile(" % ([0-9]*)\n")


def detect_ids_from_tex(tex: str) -> OrderedSet[int]:
  res = OrderedSet()
  matches = re.findall(IDS_TEX_PATTERN, tex)
  for match in matches:
    res.add(int(match))
  return res
