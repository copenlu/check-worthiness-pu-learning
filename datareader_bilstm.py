import os
import pandas as pd
import torch
import re
import glob
import json
import numpy as np
from torch.utils.data import Dataset
from typing import AnyStr, Tuple, List, Callable
from transformers import PreTrainedTokenizer
import html
from fasttext import tokenize


def text_to_batch_bilstm(text: List, tokenizer) -> Tuple[List, List]:
    """
    Creates a tokenized batch for input to a bilstm model
    :param text: A list of sentences to tokenize
    :param tokenizer: A tokenization function to use (i.e. fasttext)
    :return: Tokenized text as well as the length of the input sequence
    """
    # Some light preprocessing
    input_ids = [tokenizer.encode(t) for t in text]

    return input_ids, [len(ids) for ids in input_ids]


def collate_batch_bilstm(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    seq_lens = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]

    assert (all(len(i) == max_length for i in input_ids))
    return torch.tensor(input_ids), torch.tensor(seq_lens), torch.tensor(labels)


def collate_batch_bilstm_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_bilstm(input_data) + ([i[-1] for i in input_data],)


def collate_batch_bilstm_with_weight(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return collate_batch_bilstm(input_data) + (torch.tensor([i[3] for i in input_data]),)


def get_first_sentence_redi_et_al(text: AnyStr) -> AnyStr:
    """Applies the first sentence selector used in Redi et al (2019), code is here:
    https://github.com/mirrys/citation-needed-paper/blob/d5023eca274623963522e4a64364b572547fc014/run_citation_need_model.py#L41

    :param text: The original statement
    :return: The first sentence in the statement
    """
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]
    return text


class FasttextTokenizer:

    def __init__(self, vocabulary_file):
        self.vocab = {}
        with open(vocabulary_file) as f:
            for j,l in enumerate(f):
                self.vocab[l.strip()] = j

    def encode(self, text):
        tokens = tokenize(text.lower().replace('\n', ' ') + '\n')
        return [self.vocab[t] if t in self.vocab else self.vocab['<unk/>'] for t in tokens]


class PHEMEClassifierDataset(Dataset):
    """Datareader for basic PHEME classification with no context

    """
    def __init__(self, pheme_directory, tokenizer):
        """

        :param pheme_directory: The root directory of the PHEME data
        """
        super(PHEMEClassifierDataset, self).__init__()

        rumours = []
        non_rumours = []
        self.name = pheme_directory.split('/')[-1].split('-')[0]

        for source_tweet_file in glob.glob(f'{pheme_directory}/non-rumours/**/source-tweets/*.json'):
            with open(source_tweet_file) as js:
                tweet = json.load(js)
            non_rumours.append(tweet['text'])
        for source_tweet_file in glob.glob(f'{pheme_directory}/rumours/**/source-tweets/*.json'):
            with open(source_tweet_file) as js:
                tweet = json.load(js)
                rumours.append(tweet['text'])

        self.dataset = pd.DataFrame(rumours + non_rumours, columns=['statement'])
        self.dataset['label'] = [1] * len(rumours) + [0] * len(non_rumours)
        self.dataset['statement'] = self.dataset['statement'].str.normalize('NFKD')

        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_bilstm([row[0]], self.tokenizer)
        label = row[1]
        return input_ids, mask, label, item
