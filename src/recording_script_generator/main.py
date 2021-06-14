from collections import OrderedDict
from logging import getLogger

from speech_dataset_parser import parse_ljs
from text_selection import greedy_ngrams_epochs
from text_utils import EngToIpaMode, Language, text_to_ipa, text_to_sentences
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import text_normalize, text_to_symbols
from tqdm import tqdm

ljs = parse_ljs(
  dir_path="/data/datasets/LJSpeech-1.1",
  logger=getLogger(__name__),
)

whole_texts = str.join(" ", [item.text for item in ljs.items()])

sents = text_to_sentences(whole_texts, lang=Language.ENG, logger=getLogger(__name__))

print(sents[:10])

sents = sents[:10]

norm_sents = []
for sent in tqdm(sents):
  norm_sent = text_normalize(
    text=sent,
    lang=Language.ENG,
    logger=getLogger(__name__),
  )
  norm_sents.append(norm_sent)


ipa_sents = []
for norm_sent in tqdm(norm_sents):
  ipa = text_to_ipa(
    text=norm_sent,
    lang=Language.ENG,
    mode=EngToIpaMode.BOTH,
    replace_unknown_with="_",
    logger=getLogger(__name__),
    use_cache=True
  )
  ipa_sents.append(ipa)

settings = IPAExtractionSettings(
  ignore_arcs=True,
  ignore_tones=True,
  replace_unknown_ipa_by="_",
)

ipa_sent_symbols = OrderedDict()
for i, ipa_sent in enumerate(ipa_sents):
  ipa_symbols = text_to_symbols(
    text=ipa_sent,
    lang=Language.IPA,
    ipa_settings=settings,
    logger=getLogger(__name__),
  )
  ipa_sent_symbols[i] = ipa_symbols

res = greedy_ngrams_epochs(
  data=ipa_sent_symbols,
  n_gram=1,
  epochs=3,
  ignore_symbols=set(),
)

for i in res:
  print(sents[i])

