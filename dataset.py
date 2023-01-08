from torch.utils.data import Dataset
from os.path import exists
import wget
from nltk.tokenize import word_tokenize


class AliceInTheWonderlandDataset(Dataset):
    def __init__(self, num_words):
        dataset_path = "alice_in_wonderland.txt"
        dataset_download_url = "https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/"
        if not exists(dataset_path):
            print("Dataset is not found, downloading")
            wget.download(dataset_download_url, dataset_path)
        self.alice_strings = None
        with open(dataset_path) as f:
            self.alice_strings = "".join(f.readlines())
        self.alice_strings = self.preprocess(self.alice_strings)
        self.alice_strings = word_tokenize(self.alice_strings)
        self.vocabularies = sorted(list(set(self.alice_strings)))
        self.num_words = num_words

    def preprocess(self, text):
        text = text.lower()
        text = text.replace("(", "").replace(")", "")
        text = ' '.join(text.split())
        return text

    def sentence_to_index(self,text):
        return [self.vocabularies.index(word) for word in text]

    def index_to_sentence(self, indexes):
        return [self.vocabularies[idx] for idx in indexes]