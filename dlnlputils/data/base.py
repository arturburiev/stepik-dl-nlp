import collections
import re

import numpy as np

TOKEN_RE = re.compile(r"[\w\d]+")


def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def character_tokenize(txt):
    return list(txt)


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


def add_fake_token(word2id, token="<PAD>"):
    word2id_new = {token: i + 1 for token, i in word2id.items()}
    word2id_new[token] = 0
    return word2id_new


def texts_to_token_ids(tokenized_texts, word2id):
    return [
        [word2id[token] for token in text if token in word2id]
        for text in tokenized_texts
    ]


def build_vocabulary(
    tokenized_texts,
    max_size=1000000,
    max_doc_freq=0.8,
    min_count=5,
    pad_word=None,
    sublinear_df=False,
    smooth_df=False,
):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    # убрать слишком редкие и слишком частые слова
    word_counts = {
        word: cnt
        for word, cnt in word_counts.items()
        if cnt >= min_count and cnt / doc_n <= max_doc_freq
    }

    # отсортировать слова по убыванию частоты
    sorted_word_counts = sorted(
        word_counts.items(), reverse=True, key=lambda pair: pair[1]
    )

    # добавим несуществующее слово с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # если у нас по прежнему слишком много слов, оставить только max_size самых частотных
    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # нумеруем слова
    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    # нормируем частоты слов (получаем вектор DF)
    word2freq = []

    for _, cnt in sorted_word_counts:
        cnt_cont = cnt
        doc_n_cont = doc_n

        if smooth_df:
            cnt_cont += 1
            doc_n_cont += 1

        word2freq.append(cnt_cont / doc_n_cont)

    word2freq = np.array(word2freq, dtype="float32")

    if sublinear_df:
        word2freq = np.log(word2freq) + 1

    return word2id, word2freq


PAD_TOKEN = "__PAD__"
NUMERIC_TOKEN = "__NUMBER__"
NUMERIC_RE = re.compile(r"^([0-9.,e+\-]+|[mcxvi]+)$", re.I)


def replace_number_nokens(tokenized_texts):
    return [
        [token if not NUMERIC_RE.match(token) else NUMERIC_TOKEN for token in text]
        for text in tokenized_texts
    ]


def generate_tokens_n_grams(tokens, ngram_range):
    assert len(ngram_range) == 2, "The ngram range must be a tuple of two elements"
    range_start, range_end = ngram_range[0], ngram_range[1] + 1
    result = []

    for i in range(range_start, range_end):
        result += [
            " ".join(tokens_ngram.tolist())
            for tokens_ngram in np.lib.stride_tricks.sliding_window_view(
                tokens, window_shape=i
            )
        ]

    return result
