from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from itertools import islice
from logging import getLogger
from time import perf_counter, sleep
from typing import Any, List
from typing import OrderedDict as OrderedDictType

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from tqdm import tqdm


def get_chunked_dict_keys(dictionary: OrderedDictType[Any, Any], chunk_size: int) -> List[OrderedSet[Any]]:
  logger = getLogger(__name__)
  logger.info("Creating chunks...")
  now = perf_counter()
  keys = OrderedSet(dictionary.keys())
  #key_chunks = [keys[i:i + chunk_size] for i in tqdm(range(0, len(keys), chunk_size))]
  iterator = iter(keys)
  result = list(iter(lambda: list(islice(iterator, chunk_size)), list()))
  duration = perf_counter() - now
  logger.info(f"Done. Duration: {duration:.2f}s.")
  return result

  # return key_chunks


def get_chunk(chunked_keys: OrderedSet[UtteranceId], utterances: Utterances) -> Utterances:
  logger = getLogger(__name__)
  now = perf_counter()
  logger.info("Getting chunk...")
  result = Utterances({k: utterances[k] for k in chunked_keys})
  result.language = utterances.language
  result.symbol_format = utterances.symbol_format
  duration = perf_counter() - now
  logger.info(f"Done. Duration: {duration:.2f}s.")
  return result


class ChunkingTransformer():
  def fit(self) -> None:
    pass

  def transform(self, utterances: Utterances, chunksize: int) -> List[Utterances]:
    logger = getLogger(__name__)
    if len(utterances) <= chunksize:
      return [utterances]
    logger.info("Chunking utterances...")
    chunked_keys = get_chunked_dict_keys(utterances, chunk_size=chunksize)
    logger.info("Done.")
    logger.info("Getting chunks.")
    method = partial(get_chunk, utterances=utterances)
    # with ProcessPoolExecutor(max_workers=15) as ex:
    now = perf_counter()
    # with ThreadPoolExecutor(max_workers=15) as ex:
    #   chunks: List[Utterances] = list(tqdm(
    #     ex.map(method, chunked_keys, chunksize=1),
    #     total=len(chunked_keys)
    #   ))
    chunks = [get_chunk(chunk, utterances) for chunk in tqdm(chunked_keys)]
    logger.info(f"Done. Duration: {perf_counter() - now:.2f}s.")
    return chunks
