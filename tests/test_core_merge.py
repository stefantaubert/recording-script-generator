
from collections import OrderedDict

from ordered_set import OrderedSet
from recording_script_generator.core.merge import merge
from recording_script_generator.core.preparation import PreparationData
from text_utils.language import Language


def test_merge():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res = merge([d1])

  assert res.reading_passages == OrderedDict({0: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["a", "b"]})


def test_merge__ignores_duplicates_from_one_source():
  d1 = PreparationData(
    reading_passages=[["a", "b"], ["a", "b"]],
    representations=[["a", "b"], ["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res = merge([d1])

  assert res.reading_passages == OrderedDict({0: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["a", "b"]})


def test_merge__ignores_duplicates_from_multiple_sources():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  d2 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res = merge([d1, d2])

  assert res.reading_passages == OrderedDict({0: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["a", "b"]})


def test_merge__respects_order():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  d2 = PreparationData(
    reading_passages=[["c", "d"]],
    representations=[["c", "d"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res = merge([d2, d1])

  assert res.reading_passages == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})
