import math
import os
from os.path import exists
from pathlib import Path

import numpy as np
import torch
import wget
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import gzip


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

        raw_source_text = [self.start_token] + list(raw_source_text) + [self.end_token]
        raw_target_text = [self.start_token] + list(raw_target_text) + [self.end_token]

        idx_source_text = self.sentence_to_index(raw_source_text)
        idx_target_text = self.sentence_to_index(raw_target_text)
        return torch.tensor(idx_source_text), torch.tensor(idx_target_text)

    def __len__(self):
        return self.length

    def get_vocabulary_length(self):
        return len(self.vocabularies)


class Multi30kDatasetEN_DE(Dataset):
    def __init__(self, split="train", dataset_root="multi30k_dataset"):
        dataset_root_link = "https://github.com/multi30k/dataset/raw/master/data/task1/raw/"

        self.test_dataset_filename = ["test_2016_flickr.en.gz", "test_2017_flickr.en.gz", "test_2017_mscoco.en.gz",
                                      "test_2018_flickr.en.gz"]
        self.train_dataset_filename = ["train.en.gz"]
        self.val_dataset_filename = ["val.en.gz"]

        all_dataset_filename = {"train": self.train_dataset_filename, "test": self.test_dataset_filename,
                                "val": self.val_dataset_filename}
        self.dataset_filename = all_dataset_filename[split]
        self.dataset_links = [os.path.join(dataset_root_link, filename) for filename in self.dataset_filename]
        self.dataset_root = dataset_root
        if not self.is_downloaded():
            print("Dataset is not found in local directory")
            self.download_dataset()
            self.unzip_files([os.path.join(self.dataset_root, filename) for filename in self.dataset_filename])
        else:
            print("Dataset is found in local directory")
        english_raw_list, german_raw_list = self.read_dataset()
        english_tokenized_list = [word_tokenize(text, language="english") for text in english_raw_list]
        german_tokenized_list = [word_tokenize(text, language="german") for text in german_raw_list]
        self.start_token = "<SOS>"
        self.end_token = "<EOS>"
        self.pad_token = "<PAD>"
        english_vocab = self.get_vocab(english_tokenized_list)
        german_vocab = self.get_vocab(german_tokenized_list)
        english_max_seq = self.count_max_sequence(english_tokenized_list)
        german_max_seq = self.count_max_sequence(german_tokenized_list)

    def words2indexes(self, list_of_word, vocab):
        word2index_dict = {word: idx for idx, word in enumerate(vocab)}
        return [word2index_dict[word] for word in list_of_word]

    def get_vocab(self, tokenized_text_list):
        vocab_list = [self.pad_token, self.start_token, self.end_token]
        for tokenized_text in tokenized_text_list:
            vocab_list += tokenized_text
        vocab_list = sorted(list(set(vocab_list)))
        return vocab_list

    def download_dataset(self):
        def bar_custom(current, total, width=80):
            progress = int(current / total * 10) * "-" + (10 - int(current / total * 10)) * " "
            print(f"Downloading: [{progress}] {current / 1000000} MB/{total / 1000000} MB")

        print("Downloading dataset")
        Path(self.dataset_root).mkdir(parents=True, exist_ok=True)
        for data_url in self.dataset_links:
            wget.download(data_url, out=self.dataset_root, bar=bar_custom)

    def is_downloaded(self):
        for filename in self.dataset_filename:
            if not os.path.exists(os.path.join(self.dataset_root, filename)):
                return False
        return True

    def unzip_files(self, file_path_list):
        for path in file_path_list:
            output_path = path.replace(".gz", ".txt")
            op = open(output_path, "w")
            with gzip.open(path, "rb") as ip_byte:
                op.write(ip_byte.read().decode("utf-8"))
                op.close()

    def read_dataset(self):
        list_txt_path = [f'{self.dataset_root}/{item.replace(".gz", ".txt")}' for item in self.val_dataset_filename]
        english_strings, german_strings = self.read_en_de_txts(list_txt_path)
        return english_strings, german_strings

    def read_en_de_txts(self, list_of_path):
        german_strings = []
        english_strings = []
        for english_path in list_of_path:
            german_path = english_path.replace(".en", ".de")
            with open(english_path, "r") as f:
                english_strings += f.readlines()
            with open(german_path, "r") as f:
                german_strings += f.readlines()
        return english_strings, german_strings

    def count_max_sequence(self, tokenized_list):
        max = 0
        for text in tokenized_list:
            max = len(text) if len(text) > max else max
        return max
