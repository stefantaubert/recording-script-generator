import logging
import math
import string
from logging import getLogger
from typing import Optional

from recording_script_generator.core.types import Utterances
from sentence2pronunciation import (prepare_cache_mp,
                                    sentences2pronunciations_from_cache_mp)
from text_utils import SymbolFormat
from text_utils.pronunciation.G2p_cache import get_eng_g2p
from text_utils.pronunciation.main import get_eng_to_arpa_lookup_method
from text_utils.pronunciation.pronunciation_dict_cache import \
    get_eng_pronunciation_dict_arpa
from text_utils.symbol_format import SymbolFormat
from tqdm import tqdm


def convert_utterances_from_eng_to_arpa_inplace(utterances: Utterances, n_jobs: int, chunksize: Optional[int], batches: Optional[int], maxtasksperchild: Optional[int]) -> None:
  if len(utterances) > 0:
    first_utterance = utterances[list(utterances.keys())[0]]
    assert isinstance(first_utterance, tuple)
  logger = getLogger(__name__)
  logger.info("Loading dictionaries...")
  get_eng_g2p()
  get_eng_pronunciation_dict_arpa()
  logger.info("Done.")

  logger.info("Preparing conversion...")

  prn_logger = getLogger("text_utils.pronunciation.main")
  prn_logger.setLevel(logging.WARNING)

  if batches is None:
    assert chunksize is not None
  else:
    chunksize = math.ceil(len(utterances) / n_jobs / batches)

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
    maxtasksperchild=maxtasksperchild,
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
    maxtasksperchild=maxtasksperchild,
  )
  logger.info("Done.")

  logger.info("Updating utterances...")
  utterances.symbol_format = SymbolFormat.PHONEMES_ARPA
  for utterance_id, old_pronunciation in tqdm(utterances.items()):
    utterances[utterance_id] = sentence_pronunciations[old_pronunciation]
  logger.info("Done.")
