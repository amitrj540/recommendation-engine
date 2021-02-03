import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def stem_text(text):
    """
    PorterStemmer is used for Stemming.
    >>> stem_text(text) -> str
    """
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

def rem_stopwords(text, stop_word="english"):
    eng_stop_words = stopwords.words(stop_word)
    return " ".join([word for word in text.split() if word not in eng_stop_words])

def text_clean(text, reg_no_space="[.;:!\'?,\"()\[\]#]",
                    reg_space="(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)|(;)|(&amp)"):
    """
    Removes unwanted punctuations, symbols and HTML Tags.
    Default params :
    reg_no_space = "[.;:!\'?,\"()\[\]#]"
    reg_space = "(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)|(;)|(&amp)"
    >>>text_processing(text) -> str
    """
    no_space = re.compile(reg_no_space)
    space = re.compile(reg_space)
    preprocess = lambda text: " ".join(space.sub(" ", no_space.sub("", text.lower())).split())
    return preprocess(text)
