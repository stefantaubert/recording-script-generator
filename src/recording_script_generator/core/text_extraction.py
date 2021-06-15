from logging import getLogger
from typing import List

from speech_dataset_parser.data import PreDataList
from text_utils import EngToIpaMode, Language, text_to_ipa, text_to_sentences


def ljs_to_text(ljs_data: PreDataList) -> List[str]:
  whole_texts = str.join(" ", [item.text for item in ljs_data.items()])
  sents = text_to_sentences(whole_texts, lang=Language.ENG, logger=getLogger(__name__))
  return sents
