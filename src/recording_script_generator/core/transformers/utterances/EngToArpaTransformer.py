import logging
import string
from logging import getLogger
from typing import Optional

from recording_script_generator.core.types import Utterances, clone_utterances
from sentence2pronunciation import prepare_cache_mp
from sentence2pronunciation.core import sentences2pronunciations_from_cache_mp
from text_utils import SymbolFormat
from text_utils.pronunciation.G2p_cache import get_eng_g2p
from text_utils.pronunciation.main import get_eng_to_arpa_lookup_method
from text_utils.pronunciation.pronunciation_dict_cache import \
    get_eng_pronunciation_dict_arpa
from text_utils.symbol_format import SymbolFormat
from tqdm import tqdm


def get_utterances_from_eng_to_arpa(utterances: Utterances, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Utterances:
  logger = getLogger(__name__)
  logger.info("Loading dictionaries...")
  get_eng_g2p()
  get_eng_pronunciation_dict_arpa()
  logger.info("Done.")

  logger.info("Preparing conversion...")

  prn_logger = getLogger("text_utils.pronunciation.main")
  prn_logger.setLevel(logging.WARNING)

  sentences = set(utterances.values())
  cache = prepare_cache_mp(
    sentences=sentences,
    annotation_split_symbol=None,
    chunksize=chunksize,
    consider_annotation=False,
    get_pronunciation=get_eng_to_arpa_lookup_method(),
    ignore_case=True,
    n_jobs=n_jobs,
    split_on_hyphen=True,
    trim_symbols=set(string.punctuation),
  )
  logger.info(f"Done. Retrieved {len(cache)} unique words (incl. punctuation).")

  logger.info("Converting to ARPA...")
  sentence_pronunciations = sentences2pronunciations_from_cache_mp(
    sentences=sentences,
    cache=cache,
    annotation_split_symbol=None,
    chunksize=chunksize,
    consider_annotation=False,
    ignore_case=True,
    n_jobs=n_jobs,
  )
  logger.info("Done.")

  logger.info("Collecting utterances...")
  result = Utterances()
  result.language = utterances.language
  result.symbol_format = SymbolFormat.PHONEMES_ARPA

  for utterance_id, old_pronunciation in tqdm(utterances.items()):
    result[utterance_id] = sentence_pronunciations[old_pronunciation]
  logger.info("Done.")

  return result


class EngToArpaTransformer():
  def fit(self, utterances: Utterances, n_jobs: int, chunksize: int = 10000):
    logger = getLogger(__name__)
    logger.info("Loading dictionaries...")
    get_eng_g2p()
    get_eng_pronunciation_dict_arpa()
    logger.info("Done.")

    logger.info("Preparing conversion...")

    prn_logger = getLogger("text_utils.pronunciation.main")
    prn_logger.setLevel(logging.WARNING)

    sentences = set(utterances.values())
    cache = prepare_cache_mp(
      sentences=sentences,
      annotation_split_symbol=None,
      chunksize=chunksize,
      consider_annotation=False,
      get_pronunciation=get_eng_to_arpa_lookup_method(),
      ignore_case=True,
      n_jobs=n_jobs,
      split_on_hyphen=True,
      trim_symbols=set(string.punctuation),
    )
    logger.info(f"Done. Retrieved {len(cache)} unique words (incl. punctuation).")

    self.cache = cache
    self.n_jobs = n_jobs
    self.chunksize = chunksize
    self.sentences = sentences

  def transform(self, utterances: Utterances) -> Utterances:
    logger = getLogger(__name__)
    logger.info("Converting to ARPA...")
    sentence_pronunciations = sentences2pronunciations_from_cache_mp(
      sentences=self.sentences,
      cache=self.cache,
      annotation_split_symbol=None,
      chunksize=self.chunksize,
      consider_annotation=False,
      ignore_case=True,
      n_jobs=self.n_jobs,
    )
    logger.info("Done.")

    logger.info("Updating existing utterances...")
    result = clone_utterances(utterances)
    for utterance_id, old_pronunciation in tqdm(utterances.items()):
      result[utterance_id] = sentence_pronunciations[old_pronunciation]
    result.symbol_format = SymbolFormat.PHONEMES_ARPA
    logger.info("Done.")
    return result
