from recording_script_generator.core.text_extraction import (
    get_minimum_frequency, get_word_frequencies)


def test_get_minimum_frequency():
  word_frequencies = get_word_frequencies()

  res = get_minimum_frequency(["this", "is", "a", "test"], word_frequencies)

  assert res == 119
