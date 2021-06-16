from collections import OrderedDict

from ordered_set import OrderedSet
from recording_script_generator.core.merge import (ScriptData, Selection,
                                                   merge,
                                                   select_greedy_ngrams_epochs)
from recording_script_generator.core.preparation import PreparationData
from text_utils.language import Language


def test_merge__merges():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res, _ = merge([d1])

  assert res.reading_passages == OrderedDict({0: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["a", "b"]})


def test_merge__adds_all_keys_to_rest_selection():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res, selection = merge([d1])

  assert selection.selected == OrderedSet()
  assert selection.ignored == OrderedSet()
  assert selection.rest == OrderedSet(res.reading_passages.keys())


def test_merge__returns_script_data_and_selection():
  d1 = PreparationData(
    reading_passages=[["a", "b"]],
    representations=[["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res, selection = merge([d1])

  assert isinstance(res, ScriptData)
  assert isinstance(selection, Selection)


def test_merge__ignores_duplicates_from_one_source():
  d1 = PreparationData(
    reading_passages=[["a", "b"], ["a", "b"]],
    representations=[["a", "b"], ["a", "b"]],
    reading_passages_lang=Language.ENG,
    representations_lang=Language.ENG,
  )

  res, _ = merge([d1])

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

  res, _ = merge([d1, d2])

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

  res, _ = merge([d2, d1])

  assert res.reading_passages == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})
  assert res.representations == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})


def test_select_greedy_ngrams_epochs():
  data = ScriptData(
    reading_passages=None,
    representations=OrderedDict({
      0: ["a"],
      1: ["b"],
      2: ["a"],
    })
  )

  selection = Selection(
    selected=OrderedSet(),
    ignored=OrderedSet(),
    rest=OrderedSet(data.representations.keys()),
  )

  res = select_greedy_ngrams_epochs(
    data=data,
    selection=selection,
    n_gram=1,
    epochs=1,
    ignore_symbols=None,
  )

  assert res.selected == OrderedSet([0, 1])
  assert res.ignored == OrderedSet()
  assert res.rest == OrderedSet([2])
