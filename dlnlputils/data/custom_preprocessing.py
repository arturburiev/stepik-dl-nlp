import re
import string

import nltk
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
import spacy

nltk.download("omw-1.4")


def stopword_removing_preprocessing(texts, lang="english", tokenize=False):
    result = []
    stop_words = set(nltk.corpus.stopwords.words(lang))
    for text in texts:
        word_tokens = nltk.tokenize.word_tokenize(text) if tokenize else text
        result.append([w for w in word_tokens if w not in stop_words])
    return result


def stemming_preprocessing(
    texts, lemmatizing_mode=False, lang="english", tokenize=False
):
    result = []
    stem_func = None

    if lemmatizing_mode:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        stem_func = lemmatizer.lemmatize
    else:
        stemmer = nltk.stem.PorterStemmer()
        stem_func = stemmer.stem

    for text in texts:
        word_tokens = nltk.tokenize.word_tokenize(text) if tokenize else text
        result.append([stem_func for w in word_tokens])

    return result


def lemmatizing_preprocessing(texts, lang="english", tokenize=False):
    return stemming_preprocessing(
        texts, lemmatizing_mode=True, lang=lang, tokenize=tokenize
    )


def spellchecking_preprocessing(texts, lang="english", tokenize=False):
    result = []
    spell = SpellChecker()
    for text in texts:
        word_tokens = nltk.tokenize.word_tokenize(text) if tokenize else text
        result.append([spell.correction(w) for w in word_tokens])
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
