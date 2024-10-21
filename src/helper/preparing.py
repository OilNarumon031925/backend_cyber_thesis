import re
from pythainlp.corpus import thai_stopwords

stopwords = thai_stopwords()

def convert_labeled(text: str):
    if text == "bully":
        return 1
    return 0


def remove_punctuation(text, pattern = '[^ก-๙]'):
    return re.sub(pattern,'', text)


def remove_stopword(tokenize):
  stop_words = [w for w in  tokenize if w not in stopwords]
  return stop_words