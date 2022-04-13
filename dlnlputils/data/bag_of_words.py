import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset


def calc_counter_matrix(tokenized_texts, word2id):
    # считаем количество употреблений каждого слова в каждом документе
    result = scipy.sparse.dok_matrix(
        (len(tokenized_texts), len(word2id)), dtype="float32"
    )
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    return result


def build_class_counter_matrix(doc_counter_matrix, labels):
    result = np.matrix(np.empty((0, doc_counter_matrix.shape[-1]), "float32"))
    unique_labels = np.unique(labels)

    for unique_label in unique_labels:
        class_samples = doc_counter_matrix[np.where(labels == unique_label)]
        result = np.vstack((result, class_samples.sum(0)))

    return scipy.sparse.dok_matrix(result)


def calc_pmi(doc_counter_matrix, labels):
    result = None
    class_counter_matrix = build_class_counter_matrix(doc_counter_matrix, labels)
    all_sum = class_counter_matrix.sum()
    probs_w_c = class_counter_matrix / all_sum
    probs_c = class_counter_matrix.sum(1) / all_sum
    probs_w = class_counter_matrix.sum(0) / all_sum
    result = scipy.sparse.csr_matrix(np.log((probs_w_c / (probs_c * probs_w)))).max(0)
    return result


def vectorize_texts(
    doc_counter_matrix, word2freq, pmi_vec, mode="tfidf", sublinear_tf=False, scale=True
):
    def calc_tf(matrix, sublinear_mode):
        tf_matrix = matrix.tocsr()
        tf_matrix = tf_matrix.multiply(1 / tf_matrix.sum(1))

        if sublinear_mode:
            tf_matrix.data = 1 + np.log(tf_matrix.data)

        return tf_matrix

    assert mode in {"tfidf", "idf", "tf", "bin", "pmi", "tfpmi"}

    result = doc_counter_matrix.copy()

    # получаем бинарные вектора "встречается или нет"
    if mode == "bin":
        result = (result > 0).astype("float32")

    # получаем вектора относительных частот слова в документе
    elif mode == "tf":
        result = calc_tf(result, sublinear_tf)

    # полностью убираем информацию о количестве употреблений слова в данном документе,
    # но оставляем информацию о частотности слова в корпусе в целом
    elif mode == "idf":
        result = (result > 0).astype("float32").multiply(1 / word2freq)

    # учитываем всю информацию, которая у нас есть:
    # частоту слова в документе и частоту слова в корпусе
    elif mode == "tfidf":
        result = calc_tf(result, sublinear_tf)
        result = result.multiply(1 / word2freq)  # разделить каждый столбец на вес слова

    elif mode == "pmi":
        result = (result > 0).astype("float32").multiply(pmi_vec)

    elif mode == "tfpmi":
        result = calc_tf(result, sublinear_tf)
        result = result.multiply(pmi_vec)

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= result.max() + 1e-6

    return result.tocsr()


class SparseFeaturesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label
