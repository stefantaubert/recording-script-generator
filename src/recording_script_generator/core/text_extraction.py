import re
from collections import Counter
from typing import List, Set

import enchant
from text_utils import Language, SymbolFormat, text_to_sentences


def file_to_utterances(content: str, lang: Language, text_format: SymbolFormat) -> List[str]:
  content_lines = content.split("\n")
  res = []
  for line in content_lines:
    sentences = text_to_sentences(
      text=line.strip(),
      lang=lang,
      text_format=text_format,
    )
    res.extend(sentences)
  return res


def ends_with_punctuation(sentence: str) -> bool:
  if len(sentence) == 0:
    return False

  last_letter = sentence[-1]

  contains_sentence_ending = last_letter in {".", "!", "?"}
  if not contains_sentence_ending:
    return False

  return True


def starts_with_big_letter(sentence: str) -> bool:
  if len(sentence) == 0:
    return False

  first_letter = sentence[0]
  res = first_letter.isupper()
  return res


def is_sentence(sentence: str, lang: Language, sentence_format: SymbolFormat) -> bool:
  if sentence_format == SymbolFormat.GRAPHEMES:
    if lang == Language.ENG or lang == Language.GER:
      return starts_with_big_letter(sentence) and ends_with_punctuation(sentence)
    else:
      raise Exception("Not supported!")
  elif sentence_format.is_IPA:
    return ends_with_punctuation(sentence)
  else:
    assert False

# def contains_direct_speech(sentence: str) -> bool:
#   return len(set(sentence).intersection({"\""})) > 0


# def contains_parenthesis(sentence: str) -> bool:
#   return len(set(sentence).intersection({"(", ")", "[", "]", "{", "}"})) > 0


# def contains_undesired_characters(sentence: str, undesired: Set[str]) -> bool:
#   return len(set(sentence).intersection(undesired)) > 0



def strip_punctuation(word: str):
  res = word.strip(".,;-/!?:\\—()[]{}\"'")
  return res


def strip_punctuation_words(words: List[str]) -> List[str]:
  # keep
  res = [strip_punctuation(x) for x in words]
  res = [x for x in res if x != ""]
  return res


# def get_word_frequencies() -> FreqDist:
#   tmp_path = Path(tempfile.gettempdir()) / "word_freq.pkl"
#   if tmp_path.exists():
#     with tmp_path.open(mode="rb") as f:
#       word_frequencies = pickle.load(f)
#   else:
#     from nltk import download
#     download('brown', quiet=True)
#     from nltk.corpus import brown
#     word_frequencies = FreqDist(word.lower() for sentence in brown.sents()
#                                 for word in sentence)
#     with tmp_path.open(mode="wb") as f:
#       pickle.dump(word_frequencies, f)

#   return word_frequencies
