from itertools import islice
from logging import getLogger
from time import perf_counter
from typing import Any, List
from typing import OrderedDict as OrderedDictType

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from tqdm import tqdm


class DechunkingTransformer():
  def fit(self) -> None:
    pass

  def transform(self, chunks: List[Utterances]) -> Utterances:
    assert len(chunks) > 0
    logger = getLogger(__name__)
    logger.info("Dechunking utterances...")
    first_chunk = chunks[0]
    result = Utterances()
    result.language = first_chunk.language
    result.symbol_format = first_chunk.symbol_format
    for chunk in tqdm(chunks):
      result.update(chunk)
    logger.info("Done.")
    return result
