from ordered_set import OrderedSet

# def test_detect_ids_from_tex__one_id():
#   input_tex = "\\item Nor do all men find the same things the objects of their fear, anger, repulsion, and the rest. % 1\n"

#   res = detect_ids_from_tex(input_tex)

#   assert res == OrderedSet([1])


# def test_detect_ids_from_tex__two_ids():
#   input_tex = "\\item Nor do all men find the same things the objects of their fear, anger, repulsion, and the rest. % 1\n\\item Nor do all men find the same things the objects of their fear, anger, repulsion, and the rest. % 22\n"

#   res = detect_ids_from_tex(input_tex)

#   assert res == OrderedSet([1, 22])
