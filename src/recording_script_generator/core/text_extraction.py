import pickle
import re
import tempfile
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Set

import enchant
from nltk.corpus import brown
from nltk.probability import FreqDist
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


def is_sentence(sentence: str, lang: Language) -> bool:
  if lang == Language.ENG or lang == Language.GER:
    return starts_with_big_letter(sentence) and ends_with_punctuation(sentence)
  if lang == Language.IPA:
    return ends_with_punctuation(sentence)
  return sentence != ""

# def contains_direct_speech(sentence: str) -> bool:
#   return len(set(sentence).intersection({"\""})) > 0


# def contains_parenthesis(sentence: str) -> bool:
#   return len(set(sentence).intersection({"(", ")", "[", "]", "{", "}"})) > 0


# def contains_undesired_characters(sentence: str, undesired: Set[str]) -> bool:
#   return len(set(sentence).intersection(undesired)) > 0


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


def words_contain_acronyms(words: List[str]) -> bool:
  return any(is_acronym(word) for word in words)


def is_acronym(word: str) -> bool:
  if len(word) >= 3 and word.isupper():
    return True
  return False


def contains_proper_names(sentence: str) -> bool:
  pattern = r" [A-HJ-Z]"
  matches = re.search(pattern, sentence)
  return matches is not None


def get_word_frequencies() -> FreqDist:
  tmp_path = Path(tempfile.gettempdir()) / "word_freq.pkl"
  if tmp_path.exists():
    with tmp_path.open(mode="rb") as f:
      word_frequencies = pickle.load(f)
  else:
    from nltk import download
    download('brown', quiet=True)
    from nltk.corpus import brown
    word_frequencies = FreqDist(word.lower() for sentence in brown.sents()
                                for word in sentence)
    with tmp_path.open(mode="wb") as f:
      pickle.dump(word_frequencies, f)

  return word_frequencies


def get_minimum_frequency(words: List[str], word_frequencies: Counter) -> int:
  freqs = [word_frequencies[word.lower()] for word in words]
  result = min(freqs)
  for freq, word in zip(freqs, words):
    if freq <= 1:
      #print(word, freq)
      pass
  return result


UNDESIRED = {"/", "\\", "—", ":", "@", ";", "\"",
             "(", ")", "[", "]", "{", "}", "--", "quote", "oswald"}


def remove_undesired_en_sentences(sentences: List[str]) -> List[str]:
  if len(sentences) == 0:
    return []

  logger = getLogger(__name__)
  min_words_per_sentence = 3

  d = enchant.Dict("en_US")
  selected_sentence_nrs = []
  ignored = []
  sentence: str
  words_wo_punctuation: Dict[int, List[str]] = {}

  sentences = [collapse_whitespace(sentence.strip()) for sentence in sentences]

  for sentence_nr, sentence in enumerate(tqdm(sentences)):
    words = sentence.split(" ")
    words_non_punctuation = strip_punctuation_words(words)
    words_wo_punctuation[sentence_nr] = words_non_punctuation

    if not is_sentence(sentence):
      continue

    if contains_undesired_text(sentence, undesired=UNDESIRED, ignore_case=True):
      continue

    if contains_proper_names(sentence):
      #logger.info(f"Ignored {sentence} due to prober names!")
      continue

    if words_contain_acronyms(words_non_punctuation):
      continue

    if len(words) < min_words_per_sentence:
      continue

    non_dict_words_amount = get_non_dict_words_amount(words_non_punctuation, d)
    if non_dict_words_amount > 0:
      #logger.info(f"Ignored \"{sentence}\" because of {non_dict_words_amount} non-dictionary word(s).")
      continue

    selected_sentence_nrs.append(sentence_nr)

  words_counter = Counter(word.lower() for words in words_wo_punctuation.values()
                          for word in words)

  for sentence_nr in selected_sentence_nrs:
    words_wo_punc = words_wo_punctuation[sentence_nr]
    min_freq = get_minimum_frequency(words_wo_punc, words_counter)
    if min_freq <= 1:
      # selected_sentence_nrs.remove(sentence_nr)
      pass

  selected = [sentence for sentence_nr, sentence in enumerate(
    sentences) if sentence_nr in selected_sentence_nrs]
  ignored = [sentence for sentence_nr, sentence in enumerate(
    sentences) if sentence_nr not in selected_sentence_nrs]
  selected_percent = len(selected) / len(sentences)
  logger.info(
    f"Selected {selected_percent*100:.2f}% ({len(selected)}) of all {len(sentences)} sentences.")

  assert len(selected) + len(ignored) == len(sentences)

  return selected
