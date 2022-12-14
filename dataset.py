from torch.utils.data import Dataset
from os.path import exists
import wget
from nltk.tokenize import word_tokenize
import math
import numpy as np
import torch


class AliceInTheWonderlandDataset(Dataset):
    def __init__(self, num_words):
        dataset_path = "alice_in_wonderland.txt"
        dataset_download_url = "https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/"
        if not exists(dataset_path):
            print("Dataset is not found, downloading")
            wget.download(dataset_download_url, dataset_path)
        self.raw_text = self.read_text_file(dataset_path)
        self.preprocessed_text = self.preprocess(self.raw_text)
        self.tokenized_text = word_tokenize(self.preprocessed_text)
        self.start_token = "<SOS>"
        self.end_token = "<EOS>"
        self.vocabularies = [self.start_token, self.end_token] + sorted(list(set(self.tokenized_text)))
        self.chunked_tokenized_text = self.chunks_string(self.tokenized_text, num_words)
        self.length = len(self.chunked_tokenized_text)

    def read_text_file(self, file_path):
        with open(file_path) as f:
            raw_text = "".join(f.readlines())
        return raw_text

    def chunks_string(self, text, length):
        return np.array_split(text, math.ceil(len(text) / length))

    def preprocess(self, text):
        text = text.lower()
        text = text.replace("(", "").replace(")", "")
        text = ' '.join(text.split())
        return text

    def sentence_to_index(self, text):
        return [self.vocabularies.index(word) for word in text]

    def index_to_sentence(self, indexes):
        return [self.vocabularies[idx] for idx in indexes]

    def __getitem__(self, idx):
        raw_source_text = self.chunked_tokenized_text[idx]
        raw_target_text = self.chunked_tokenized_text[(idx + 1) % self.length]

        raw_source_text = [self.start_token] + list(raw_source_text) +[self.end_token]
        raw_target_text = [self.start_token] + list(raw_target_text) +[self.end_token]

        idx_source_text = self.sentence_to_index(raw_source_text)
        idx_target_text = self.sentence_to_index(raw_target_text)
        return torch.tensor(idx_source_text), torch.tensor(idx_target_text)

    def __len__(self):
        return self.length

    def get_vocabulary_length(self):
        return len(self.vocabularies)
