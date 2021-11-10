from typing import List


def strip_punctuation(word: str):
  res = word.strip(".,;-/!?:\\—()[]{}\"'")
  return res


def strip_punctuation_words(words: List[str]) -> List[str]:
  # keep
  res = [strip_punctuation(x) for x in words]
  res = [x for x in res if x != ""]
  return res


def strip_punctuation_utterance(utterance_str: str) -> str:
  words = utterance_str.split(" ")
  words_non_punctuation = strip_punctuation_words(words)
  stripped_utterance = ' '.join(words_non_punctuation)
  return stripped_utterance

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
