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
import csv

from pu_learning import get_negative_sample_weights
from pu_learning import estimate_class_prior_probability


def text_to_batch_transformer(text: AnyStr, tokenizer: PreTrainedTokenizer, text_pair: AnyStr = None) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    if text_pair is None:
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=tokenizer.max_len) for t in text]
    else:
        input_ids = [tokenizer.encode(t, text_pair=p, add_special_tokens=True, max_length=tokenizer.max_len) for t,p in zip(text, text_pair)]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels)


def collate_batch_transformer_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)


def collate_batch_transformer_with_weight(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return collate_batch_transformer(input_data) + (torch.tensor([i[3] for i in input_data]),)


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


class WikipediaCitationDataset(Dataset):
    """Dataset reader for citation_detection citation dataset

    """

    def __init__(self, pos_data_loc: AnyStr, neg_data_loc: AnyStr, tokenizer: PreTrainedTokenizer):
        super(WikipediaCitationDataset, self).__init__()
        self.pos_data = pd.read_csv(pos_data_loc, sep='\t').fillna('')
        self.neg_data = pd.read_csv(neg_data_loc, sep='\t').fillna('')
        self.tokenizer = tokenizer

        # Combine into one dataset w/ labels
        self.dataset = pd.concat([self.pos_data, self.neg_data], axis=0, ignore_index=True)[['statement']]
        # Normalize the strings
        self.dataset['statement'] = self.dataset['statement'].str.normalize('NFKD')
        # Extract the first sentence
        self.dataset['statement'] = self.dataset['statement'].apply(get_first_sentence_redi_et_al)
        # Add labels
        self.dataset['label'] = [1] * self.pos_data.shape[0] + [0] * self.neg_data.shape[0]

        self.dataset = self.dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        return input_ids, mask, label, item

class PULearningWikipediaCitationDataset(Dataset):
    """Dataset reader for citation detection with positive unlabelled learning

    """

    def __init__(
            self,
            base_dataset: Dataset,
            validation_dataset: Dataset,
            base_network: torch.nn.Module,
            device: torch.device,
            scale: int = 1.0
    ):

        super(PULearningWikipediaCitationDataset, self).__init__()
        # Subset is used as the dataset
        if type(base_dataset) == torch.utils.data.Subset:
            self.tokenizer = base_dataset.dataset.tokenizer
            indices = base_dataset.indices
            orig_dataset = base_dataset.dataset.dataset.copy()
            base_dataset = base_dataset.dataset
            base_dataset.dataset = orig_dataset.iloc[indices]
            base_dataset.dataset = base_dataset.dataset.reset_index(drop=True)

        # Only look at negative samples
        original_dataset = base_dataset.dataset.copy()
        base_dataset.dataset = base_dataset.dataset[base_dataset.dataset['label'] == 0]
        # Set the label to 1
        base_dataset.dataset['label'] = [1]*base_dataset.dataset.shape[0]

        # Get negatives weight, combine into one dataset and duplicate the negatives
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        neg_weights = get_negative_sample_weights(train_dl, val_dl, base_network, device)
        assert neg_weights.shape == base_dataset.dataset.shape, "Should have double the number of negative sample weights"
        weights = np.asarray([0.] * original_dataset.shape[0])
        weights[original_dataset.index[original_dataset['label'] == 1].tolist()] = 1.
        weights[original_dataset.index[original_dataset['label'] == 0].tolist()] = neg_weights[:,0]
        original_dataset['weight'] = weights
        duplicated_data = base_dataset.dataset.copy()
        duplicated_data['weight'] = neg_weights[:,1]
        self.dataset = pd.concat([original_dataset, duplicated_data], ignore_index=True)
        self.scale = scale


    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        weight = self.scale * row[2]
        return input_ids, mask, label, weight, item


class PULearningPriorBasedConversionWikipediaCitationDataset(Dataset):
    """Dataset reader for citation detection with positive unlabelled learning and
    positive-negative removal

    """

    def __init__(
            self,
            base_dataset: Dataset,
            validation_dataset: Dataset,
            base_network: torch.nn.Module,
            device: torch.device,
            gamma: float,
            scale: int = 1.0
    ):
        super(PULearningPriorBasedConversionWikipediaCitationDataset, self).__init__()
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        prior = estimate_class_prior_probability(base_network, train_dl, val_dl, device)
        print(prior)
        # Subset is used as the dataset
        if type(base_dataset) == torch.utils.data.Subset:
            self.tokenizer = base_dataset.dataset.tokenizer
            indices = base_dataset.indices
            orig_dataset = base_dataset.dataset.dataset.copy()
            base_dataset = base_dataset.dataset
            base_dataset.dataset = orig_dataset.iloc[indices]
            base_dataset.dataset = base_dataset.dataset.reset_index(drop=True)

        # Only look at negative samples
        original_dataset = base_dataset.dataset.copy()
        base_dataset.dataset = base_dataset.dataset[base_dataset.dataset['label'] == 0]

        # Get negatives weight, combine into one dataset and duplicate the negatives
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        neg_weights = get_negative_sample_weights(train_dl, val_dl, base_network, device)
        assert neg_weights.shape == base_dataset.dataset.shape, "Should have double the number of negative sample weights"

        positives = original_dataset[original_dataset['label'] == 1]
        positives['weight'] = [1.] * positives.shape[0]
        # Keep adding examples until p(y=1) equals our estimate
        keep_examples = np.asarray([True] * neg_weights.shape[0])
        ordered_idx = np.argsort(neg_weights[:,1])[::-1]
        i = 0
        while (positives.shape[0] + sum(~keep_examples)) / original_dataset.shape[0] < prior:
            keep_examples[ordered_idx[i]] = False
            i += 1
        kept_negatives = base_dataset.dataset[keep_examples].copy()
        kept_negatives_plus = kept_negatives.copy()
        kept_negatives_plus['label'] = [1] * kept_negatives_plus.shape[0]
        kept_negatives['weight'] = neg_weights[keep_examples, 0]
        kept_negatives_plus['weight'] = neg_weights[keep_examples, 1]
        converted_positives = base_dataset.dataset[~keep_examples].copy()
        converted_positives['label'] = [1] * converted_positives.shape[0]
        converted_positives['weight'] = [1.] * converted_positives.shape[0]

        print(positives.shape)
        print(kept_negatives.shape)
        print(converted_positives.shape)
        self.dataset = pd.concat([positives, kept_negatives, kept_negatives_plus, converted_positives],
                                 ignore_index=True)
        self.scale = scale

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        weight = self.scale * row[2]
        return input_ids, mask, label, weight, item


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
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        return input_ids, mask, label, item

    def get_row(self, row):
        return self.dataset[row]


class PULearningPHEMEDataset(Dataset):
    """Dataset reader for citation detection with positive unlabelled learning

    """

    def __init__(
            self,
            base_dataset: Dataset,
            validation_dataset: Dataset,
            base_network: torch.nn.Module,
            device: torch.device,
            scale: int = 1.0
    ):
        super(PULearningPHEMEDataset, self).__init__()
        # Subset is used as the dataset
        if type(base_dataset) == torch.utils.data.Subset:
            self.tokenizer = base_dataset.dataset.tokenizer
            indices = base_dataset.indices
            orig_dataset = base_dataset.dataset.dataset.copy()
            base_dataset = base_dataset.dataset
            base_dataset.dataset = orig_dataset.iloc[indices]
            base_dataset.dataset = base_dataset.dataset.reset_index(drop=True)
        else:
            self.tokenizer = base_dataset.tokenizer

        # Only look at negative samples
        original_dataset = base_dataset.dataset.copy()
        base_dataset.dataset = base_dataset.dataset[base_dataset.dataset['label'] == 0]
        # Set the label to 1
        base_dataset.dataset['label'] = [1] * base_dataset.dataset.shape[0]

        # Get negatives weight, combine into one dataset and duplicate the negatives
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        neg_weights = get_negative_sample_weights(train_dl, val_dl, base_network, device)
        assert neg_weights.shape == base_dataset.dataset.shape, "Should have double the number of negative sample weights"
        weights = np.asarray([0.] * original_dataset.shape[0])
        weights[original_dataset.index[original_dataset['label'] == 1].tolist()] = 1.
        weights[original_dataset.index[original_dataset['label'] == 0].tolist()] = neg_weights[:, 0]
        original_dataset['weight'] = weights
        duplicated_data = base_dataset.dataset.copy()
        duplicated_data['weight'] = neg_weights[:, 1]
        self.dataset = pd.concat([original_dataset, duplicated_data], ignore_index=True)
        self.scale = scale

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        weight = self.scale * row[2]
        return input_ids, mask, label, weight, item


class PULearningPriorBasedConversionPHEMEDataset(Dataset):
    """Dataset reader for citation detection with positive unlabelled learning and
    positive-negative removal

    """

    def __init__(
            self,
            base_dataset: Dataset,
            validation_dataset: Dataset,
            base_network: torch.nn.Module,
            device: torch.device,
            gamma: float = 1.0,
            scale: int = 1.0
    ):
        super(PULearningPriorBasedConversionPHEMEDataset, self).__init__()
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        prior = estimate_class_prior_probability(base_network, train_dl, val_dl, device)
        print(prior)
        # Subset is used as the dataset
        if type(base_dataset) == torch.utils.data.Subset:
            self.tokenizer = base_dataset.dataset.tokenizer
            indices = base_dataset.indices
            orig_dataset = base_dataset.dataset.dataset.copy()
            base_dataset = base_dataset.dataset
            base_dataset.dataset = orig_dataset.iloc[indices]
            base_dataset.dataset = base_dataset.dataset.reset_index(drop=True)
        else:
            self.tokenizer = base_dataset.tokenizer

        # Only look at negative samples
        original_dataset = base_dataset.dataset.copy()
        base_dataset.dataset = base_dataset.dataset[base_dataset.dataset['label'] == 0]

        # Get negatives weight, combine into one dataset and duplicate the negatives
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_batch_transformer)
        neg_weights = get_negative_sample_weights(train_dl, val_dl, base_network, device)
        assert neg_weights.shape == base_dataset.dataset.shape, "Should have double the number of negative sample weights"

        positives = original_dataset[original_dataset['label'] == 1]
        positives['weight'] = [1.] * positives.shape[0]
        # Keep adding examples until p(y=1) equals our estimate
        keep_examples = np.asarray([True] * neg_weights.shape[0])
        ordered_idx = np.argsort(neg_weights[:,1])[::-1]
        i = 0
        while (positives.shape[0] + sum(~keep_examples)) / original_dataset.shape[0] < prior:
            keep_examples[ordered_idx[i]] = False
            i += 1
        kept_negatives = base_dataset.dataset[keep_examples].copy()
        kept_negatives_plus = kept_negatives.copy()
        kept_negatives_plus['label'] = [1] * kept_negatives_plus.shape[0]
        kept_negatives['weight'] = neg_weights[keep_examples, 0]
        kept_negatives_plus['weight'] = neg_weights[keep_examples, 1]
        converted_positives = base_dataset.dataset[~keep_examples].copy()
        converted_positives['label'] = [1] * converted_positives.shape[0]
        converted_positives['weight'] = [1.] * converted_positives.shape[0]

        print(positives.shape)
        print(kept_negatives.shape)
        print(converted_positives.shape)
        self.dataset = pd.concat([positives, kept_negatives, kept_negatives_plus, converted_positives],
                                 ignore_index=True)
        self.scale = scale

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        weight = self.scale * row[2]
        return input_ids, mask, label, weight, item


class ClefClassifierDataset(Dataset):
        """Datareader for basic Clef classification with no context

        """

        def __init__(self, loc, tokenizer, name='clef_data'):
            """

            :param loc: The root directory of the PHEME data
            """
            super(ClefClassifierDataset, self).__init__()

            data = []
            if os.path.isdir(loc):
                files = glob.glob(f'{loc}/*.tsv') + glob.glob(f'{loc}/*.txt')
            else:
                files = [loc]

            for file in files:
                with open(file) as f:
                    data.extend([l.strip().split('\t')[-2:] for l in f])

            self.dataset = pd.DataFrame(data, columns=['statement', 'label'])
            self.dataset['statement'] = self.dataset['statement'].str.normalize('NFKD')
            self.dataset['label'] = pd.to_numeric(self.dataset['label'])

            self.tokenizer = tokenizer
            self.name = name

        def __len__(self):
            return self.dataset.shape[0]

        def __getitem__(self, item) -> Tuple:
            row = self.dataset.values[item]
            input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
            label = row[1]
            return input_ids, mask, label, item
