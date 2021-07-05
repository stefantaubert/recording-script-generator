from logging import getLogger
from typing import List, Set

import enchant
from text_utils import Language, text_to_sentences
from text_utils.text import sentence_to_words


def file_to_text(content: str, lang: Language) -> List[str]:
  content_lines = content.split("\n")
  res = []
  for line in content_lines:
    sentences = text_to_sentences(line.strip(), lang=lang, logger=getLogger(__name__))
    res.extend(sentences)
  return res


def is_relevant_sentence(sentence: str) -> bool:
  if sentence == "":
    return False
  last_letter = sentence[-1]

  contains_sentence_ending = last_letter in {".", "!", "?"}
  if not contains_sentence_ending:
    return False

  if not starts_with_big_letter(sentence):
    return False

  return True


def starts_with_big_letter(sentence: str) -> bool:
  if len(sentence) == 0:
    return False

  first_letter = sentence[0]
  res = first_letter.isupper()
  return res


def contains_direct_speech(sentence: str) -> bool:
  return len(set(sentence).intersection({"\""})) > 0


def contains_parenthesis(sentence: str) -> bool:
  return len(set(sentence).intersection({"(", ")", "[", "]", "{", "}"})) > 0


def contains_undesired_characters(sentence: str, undesired: Set[str]) -> bool:
  return len(set(sentence).intersection(undesired)) > 0


def contains_undesired_text(sentence: str, undesired: Set[str]) -> bool:
  for x in undesired:
    if x in sentence:
      return True
  return False


def get_known_words_ratio(words: List[str], dict: enchant.Dict) -> float:
  tmp = []
  for word in words:
    if word == "":
      continue
    val = int(dict.check(word))
    tmp.append(val)
  res = sum(tmp) / len(tmp)
  return res


def strip_punctuation(word: str):
  res = word.strip(".,;-/!?:\\—()[]{}")
  return res


def strip_punctuation_words(words: List[str]) -> List[str]:
  res = [strip_punctuation(x) for x in words]
  return res


def words_contain_any_with_only_capital_letters(words: List[str]) -> bool:
  for word in words:
    if len(word) >= 3 and word.isupper():
      return True
  return False


def remove_non_sentences(sentences: List[str]) -> List[str]:
  if len(sentences) == 0:
    return []

  logger = getLogger(__name__)

  d = enchant.Dict("en_US")
  res = []
  ignored = []
  for i, sentence in enumerate(sentences):

    if not is_relevant_sentence(sentence):
      continue

    if contains_direct_speech(sentence):
      continue

    if contains_parenthesis(sentence):
      continue

    if contains_undesired_characters(sentence, undesired={"/", "\\", "—", ":"}):
      continue

    if contains_undesired_text(sentence, undesired={"--"}):
      continue

    words = sentence.split(" ")
    if len(words) < 7:
      continue

    words_non_punctuation = strip_punctuation_words(words)

    if words_contain_any_with_only_capital_letters(words_non_punctuation):
      continue

    ratio = get_known_words_ratio(words_non_punctuation, d)
    if ratio < 1.0:
      logger.info(f"Ignored \"{sentence}\" because of non-dictionary words.")
      continue

    res.append(i)

  selected = [x for i, x in enumerate(sentences) if i in res]
  ignored = [x for i, x in enumerate(sentences) if i not in res]
  selected_percent = len(selected) / len(sentences)
  logger.info(
    f"Selected {selected_percent*100:.2f}% ({len(selected)}) of all {len(sentences)} sentences.")

  assert len(selected) + len(ignored) == len(sentences)

  return selected
