# from collections import OrderedDict

# import numpy as np
# from ordered_set import OrderedSet
# from text_utils.language import Language


# def test_merge__merges():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, _ = merge([d1])

#   assert res.reading_passages == OrderedDict({0: ["a", "b"]})
#   assert res.representations == OrderedDict({0: ["a", "b"]})


# def test_merge__adds_all_keys_to_rest_selection():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, selection = merge([d1])

#   assert selection.selected == OrderedSet()
#   assert selection.ignored == OrderedSet()
#   assert selection.rest == OrderedSet(res.reading_passages.keys())


# def test_merge__returns_script_data_and_selection():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, selection = merge([d1])

#   assert isinstance(res, ScriptData)
#   assert isinstance(selection, Selection)


# def test_merge__ignores_duplicates_from_one_source():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"], ["a", "b"]],
#     representations=[["a", "b"], ["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, _ = merge([d1])

#   assert res.reading_passages == OrderedDict({0: ["a", "b"]})
#   assert res.representations == OrderedDict({0: ["a", "b"]})


# def test_merge__ignores_duplicates_from_multiple_sources():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   d2 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, _ = merge([d1, d2])

#   assert res.reading_passages == OrderedDict({0: ["a", "b"]})
#   assert res.representations == OrderedDict({0: ["a", "b"]})


# def test_merge__respects_order():
#   d1 = PreparationData(
#     reading_passages=[["a", "b"]],
#     representations=[["a", "b"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   d2 = PreparationData(
#     reading_passages=[["c", "d"]],
#     representations=[["c", "d"]],
#     reading_passages_lang=Language.ENG,
#     representations_lang=Language.ENG,
#   )

#   res, _ = merge([d2, d1])

#   assert res.reading_passages == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})
#   assert res.representations == OrderedDict({0: ["c", "d"], 1: ["a", "b"]})


# def test_select_greedy_ngrams_epochs():
#   data = ScriptData(
#     reading_passages=None,
#     representations=OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["a"],
#     })
#   )

#   selection = Selection(
#     selected=OrderedSet(),
#     ignored=OrderedSet(),
#     rest=OrderedSet(data.representations.keys()),
#   )

#   res = select_greedy_ngrams_epochs(
#     data=data,
#     selection=selection,
#     n_gram=1,
#     epochs=1,
#     ignore_symbols=None,
#   )

#   assert res.selected == OrderedSet([0, 1])
#   assert res.ignored == OrderedSet()
#   assert res.rest == OrderedSet([2])


# def test_ignore():
#   data = ScriptData(
#     reading_passages=None,
#     representations=OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["a"],
#     })
#   )

#   selection = Selection(
#     selected=OrderedSet(),
#     ignored=OrderedSet(),
#     rest=OrderedSet(data.representations.keys()),
#   )

#   res = ignore(
#     data=data,
#     selection=selection,
#     ignore_symbol="a",
#   )

#   assert res.selected == OrderedSet()
#   assert res.ignored == OrderedSet([0, 2])
#   assert res.rest == OrderedSet([1])


# def test_log_stats():
#   data = ScriptData(
#     reading_passages=OrderedDict({
#       0: ["aa"],
#       1: ["bb"],
#       2: ["aa"],
#     }),
#     representations=OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["a"],
#     })
#   )

#   selection = Selection(
#     selected=OrderedSet(),
#     ignored=OrderedSet(),
#     rest=OrderedSet(data.representations.keys()),
#   )

#   log_stats(
#     data=data,
#     selection=selection,
#     avg_chars_per_s=25,
#   )

#   assert True

# def test_select_rest():
#   selection = Selection(
#     selected=OrderedSet([4, 5]),
#     ignored=OrderedSet([6, 7]),
#     rest=OrderedSet([1, 2, 3]),
#   )

#   res = select_rest(selection)

#   assert res.selected == OrderedSet([4, 5, 1, 2, 3])
#   assert res.ignored == OrderedSet([6, 7])
#   assert res.rest == OrderedSet()


# def test_number_prepend_zeros__zero():
#   res = number_prepend_zeros(0, 0)
#   assert res == "0"


# def test_number_prepend_zeros__one():
#   res = number_prepend_zeros(5, 6)
#   assert res == "5"


# def test_number_prepend_zeros__two():
#   res = number_prepend_zeros(5, 10)
#   assert res == "05"


# def test_number_prepend_zeros__three():
#   res = number_prepend_zeros(5, 100)
#   assert res == "005"


# def test_get_reading_scripts():
#   data = ScriptData(
#     reading_passages=OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["a"],
#     }),
#     representations=OrderedDict({
#       0: ["aa"],
#       1: ["bb"],
#       2: ["cc"],
#     })
#   )

#   selection = Selection(
#     selected=OrderedSet([0]),
#     ignored=OrderedSet([1]),
#     rest=OrderedSet([2]),
#   )

#   selected_df, ignored_df, rest_df = get_reading_scripts(data, selection)

#   assert list(selected_df["id"]) == [0]
#   assert list(ignored_df["id"]) == [1]
#   assert list(rest_df["id"]) == [2]


# def test_merge_merged():
#   data = ScriptData(
#     reading_passages=OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["c"],
#     }),
#     representations=OrderedDict({
#       0: ["aa"],
#       1: ["bb"],
#       2: ["cc"],
#     })
#   )

#   selection = Selection(
#     selected=OrderedSet([0]),
#     ignored=OrderedSet([1]),
#     rest=OrderedSet([2]),
#   )

#   res_data, res_selection = merge_merged([(data, selection), (data, selection)])

#   assert res_data.reading_passages == OrderedDict({
#       0: ["a"],
#       1: ["b"],
#       2: ["c"],
#       3: ["a"],
#       4: ["b"],
#       5: ["c"],
#   })
#   assert res_data.representations == OrderedDict({
#       0: ["aa"],
#       1: ["bb"],
#       2: ["cc"],
#       3: ["aa"],
#       4: ["bb"],
#       5: ["cc"],
#   })
#   assert res_selection.selected == OrderedSet([0, 3])
#   assert res_selection.ignored == OrderedSet([1, 4])
#   assert res_selection.rest == OrderedSet([2, 5])


# def test_get_df_from_reading_passages():
#   reading_passages = OrderedDict({
#     0: (["a"], ["aa"]),
#     1: (["b"], ["bb"]),
#     9: (["c"], ["cc"]),
#   })

#   res = get_df_from_reading_passages(reading_passages)

#   assert len(res) == 3
#   assert list(res.columns) == ["id", "nr", "utterance", "representation"]
#   assert list(res["id"]) == [0, 1, 9]
#   assert list(res["nr"]) == ["1", "2", "3"]
#   assert list(res["utterance"]) == ["a", "b", "c"]
#   assert list(res["representation"]) == ["aa", "bb", "cc"]
#   assert list(res.dtypes) == [np.int64, object, object, object]
