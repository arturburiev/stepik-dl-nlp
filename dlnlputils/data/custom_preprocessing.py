import re
import string

import nltk
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
import spacy

nltk.download("omw-1.4")


def lowercase_preprocessing(texts):
    return [text.lower() for text in texts]


def punctuation_removing_preprocessing(texts):
    return [text.translate(str.maketrans("", "", string.punctuation)) for text in texts]


def html_removing_preprocessing(texts):
    regular_expr = r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"
    return [re.sub(regular_expr, "", text) for text in texts]


def stopword_removing_preprocessing(tokenized_texts, lang="english"):
    result = []
    stop_words = set(nltk.corpus.stopwords.words(lang))
    for tokenized_text in tokenized_texts:
        result.append([w for w in tokenized_text if w not in stop_words])
    return result


def stemming_preprocessing(tokenized_texts, lemmatizing_mode=False, lang="english"):
    result = []
    stem_func = None

    if lemmatizing_mode:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        stem_func = lemmatizer.lemmatize
    else:
        stemmer = nltk.stem.PorterStemmer()
        stem_func = stemmer.stem

    for tokenized_text in tokenized_texts:
        result.append([stem_func(w) for w in tokenized_text])

    return result


def lemmatizing_preprocessing(tokenized_texts, lang="english"):
    return stemming_preprocessing(tokenized_texts, lemmatizing_mode=True, lang=lang)


def spellchecking_preprocessing(tokenized_texts, lang="english"):
    result = []
    spell = SpellChecker()
    for tokenized_text in tokenized_texts:
        result.append([spell.correction(w) for w in tokenized_text])
    return result


def NER_preprocessing(texts, lang="english", drop_entities=False):
    nlp = spacy.load("en_core_web_sm")
    result = []

    for text in texts:
        doc = nlp(text)

        for e in doc.ents:
            replacement = (
                ""
                if drop_entities
                else "__nertag__{}__nertag__".format(e.label_.lower())
            )
            text = text.replace(e.text, replacement)

        result.append(text)

    return result
