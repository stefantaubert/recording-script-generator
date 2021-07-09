import re
from logging import getLogger
from typing import List, Set

import enchant
from text_utils import Language, text_to_sentences
from text_utils.adjustments import collapse_whitespace
from text_utils.text import sentence_to_words
from tqdm import tqdm


def file_to_text(content: str, lang: Language) -> List[str]:
  content_lines = content.split("\n")
  res = []
  for line in content_lines:
    sentences = text_to_sentences(line.strip(), lang=lang, logger=getLogger(__name__))
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


def is_sentence(sentence: str) -> bool:
  return starts_with_big_letter(sentence) and ends_with_punctuation(sentence)


def contains_direct_speech(sentence: str) -> bool:
  return len(set(sentence).intersection({"\""})) > 0


def contains_parenthesis(sentence: str) -> bool:
  return len(set(sentence).intersection({"(", ")", "[", "]", "{", "}"})) > 0


def contains_undesired_characters(sentence: str, undesired: Set[str]) -> bool:
  return len(set(sentence).intersection(undesired)) > 0


def contains_undesired_text(sentence: str, undesired: Set[str], ignore_case: bool) -> bool:
  for x in undesired:
    if ignore_case:
      if x.lower() in sentence.lower():
        return True
    else:
      if x in sentence:
        return True

  return False


def get_non_dict_words_amount(words: List[str], dict: enchant.Dict) -> int:
  tmp = []
  for word in words:
    assert word != ""
    val = int(dict.check(word))
    tmp.append(val)
  words_in_dict = sum(tmp)
  words_not_in_dict = len(tmp) - words_in_dict
  return words_not_in_dict


def strip_punctuation(word: str):
  res = word.strip(".,;-/!?:\\—()[]{}")
  return res


def strip_punctuation_words(words: List[str]) -> List[str]:
  res = [strip_punctuation(x) for x in words]
  res = [x for x in res if x != ""]
  return res


def words_contain_any_with_only_capital_letters(words: List[str]) -> bool:
  for word in words:
    if len(word) >= 3 and word.isupper():
      return True
  return False


def contains_proper_names(sentence: str) -> bool:
  pattern = r" [A-HJ-Z]"
  matches = re.search(pattern, sentence)
  return matches is not None


def remove_undesired_en_sentences(sentences: List[str]) -> List[str]:
  if len(sentences) == 0:
    return []

  logger = getLogger(__name__)
  min_words_per_sentence = 3

  d = enchant.Dict("en_US")
  res = []
  ignored = []
  sentence: str
  for i, sentence in enumerate(tqdm(sentences)):
    sentence = collapse_whitespace(sentence.strip())

    if not is_sentence(sentence):
      continue

    if contains_direct_speech(sentence):
      continue

    if contains_parenthesis(sentence):
      continue

    if contains_undesired_characters(sentence, undesired={"/", "\\", "—", ":", "@", ";"}):
      continue

    if contains_undesired_text(sentence, undesired={"--", "quote", "oswald"}, ignore_case=True):
      continue

    if contains_proper_names(sentence):
      logger.info(f"Ignored {sentence} due to prober names!")
      continue

    words = sentence.split(" ")
    if len(words) < min_words_per_sentence:
      continue

    words_non_punctuation = strip_punctuation_words(words)

    if words_contain_any_with_only_capital_letters(words_non_punctuation):
      continue

    non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, d)
    if non_dict_words_amount > 0:
      #logger.info(f"Ignored \"{sentence}\" because of {non_dict_words_amount} non-dictionary word(s).")
      continue

    res.append(i)

  selected = [x for i, x in enumerate(sentences) if i in res]
  ignored = [x for i, x in enumerate(sentences) if i not in res]
  selected_percent = len(selected) / len(sentences)
  logger.info(
    f"Selected {selected_percent*100:.2f}% ({len(selected)}) of all {len(sentences)} sentences.")

  assert len(selected) + len(ignored) == len(sentences)

  return selected
