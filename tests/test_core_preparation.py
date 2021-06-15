
from collections import OrderedDict

from recording_script_generator.core.preparation import (PreparationData,
                                                         PreparationTarget,
                                                         add_corpus_from_text,
                                                         convert_to_ipa,
                                                         normalize)
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.language import Language
from text_utils.text import EngToIpaMode


def test_add_corpus_from_text():
  res = add_corpus_from_text(
    utterances=["test1 t", "test2 t"],
    lang=Language.ENG,
    ipa_settings=None,
  )
  assert res.reading_passages[0] == ["t", "e", "s", "t", "1", " ", "t"]
  assert res.reading_passages[1] == ["t", "e", "s", "t", "2", " ", "t"]
  assert res.representations[0] == ["t", "e", "s", "t", "1", " ", "t"]
  assert res.representations[1] == ["t", "e", "s", "t", "2", " ", "t"]
  assert res.reading_passages_lang == Language.ENG
  assert res.representations_lang == Language.ENG


def test_normalize():
  data = PreparationData(
    reading_passages=OrderedDict({
      0: ["t", " ", " ", "t"],
    }),
    reading_passages_lang=Language.ENG,
    representations=OrderedDict({
      0: ["t", " ", " ", "t"],
    }),
    representations_lang=Language.ENG,
  )

  normalize(data, target=PreparationTarget.BOTH, ipa_settings=None)

  assert data.reading_passages[0] == ["t", " ", "t"]
  assert data.representations[0] == ["t", " ", "t"]
  assert data.reading_passages_lang == Language.ENG
  assert data.representations_lang == Language.ENG


def test_convert_to_ipa():
  data = PreparationData(
    reading_passages=OrderedDict({
      0: ["t", "e", "s", "t"],
    }),
    reading_passages_lang=Language.ENG,
    representations=OrderedDict({
      0: ["t", "e", "s", "t"],
    }),
    representations_lang=Language.ENG,
  )

  convert_to_ipa(
    data=data,
    target=PreparationTarget.BOTH,
    ipa_settings=IPAExtractionSettings(
      ignore_tones=True,
      ignore_arcs=True,
      replace_unknown_ipa_by="_"
    ),
    mode=EngToIpaMode.BOTH,
    replace_unknown_with="_",
    use_cache=True,
  )

  assert data.reading_passages[0] == ["t", "ˈ", "ɛ", "s", "t"]
  assert data.representations[0] == ["t", "ˈ", "ɛ", "s", "t"]
  assert data.reading_passages_lang == Language.IPA
  assert data.representations_lang == Language.IPA
