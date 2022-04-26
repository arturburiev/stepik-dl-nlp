import torch
from torch.utils.data import TensorDataset

from dlnlputils.pipeline import predict_with_model
from .base import tokenize_corpus


def pos_corpus_to_tensor(
    sentences,
    char2id,
    label2id,
    max_sent_len,
    max_token_len,
    start_word_id=0,
    end_word_id=0,
):
    inputs = torch.zeros(
        (len(sentences), max_sent_len, max_token_len + 2), dtype=torch.long
    )
    targets = torch.zeros((len(sentences), max_sent_len), dtype=torch.long)

    for sent_i, sent in enumerate(sentences):
        for token_i, token in enumerate(sent):
            if token.form is None or token.upos is None:
                continue
            targets[sent_i, token_i] = label2id.get(token.upos, 0)

            inputs[sent_i, token_i, 0] = start_word_id

            for char_i, char in enumerate(token.form):
                inputs[sent_i, token_i, char_i + 1] = char2id.get(char, 0)

            inputs[sent_i, token_i, len(token.form) + 1] = end_word_id

    return inputs, targets


class POSTagger:
    def __init__(
        self,
        model,
        char2id,
        id2label,
        max_sent_len,
        max_token_len,
        start_word_id=0,
        end_word_id=0,
    ):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len

    def __call__(self, sentences):
        tokenized_corpus = tokenize_corpus(sentences, min_token_size=1)

        inputs = torch.zeros(
            (len(sentences), self.max_sent_len, self.max_token_len + 2),
            dtype=torch.long,
        )

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):

                inputs[sent_i, token_i, 0] = start_word_id

                for char_i, char in enumerate(token):
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

                inputs[sent_i, token_i, len(token) + 1] = end_word_id

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = predict_with_model(
            self.model, dataset
        )  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append(
                [self.id2label[cls] for cls in predicted_classes[sent_i, : len(sent)]]
            )
        return result
