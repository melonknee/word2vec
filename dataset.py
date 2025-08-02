from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset
import torch

from util import subsample


# class CBOWDataset(Dataset):

class SkipGramDataset(Dataset):
    def __init__(self, encoded_tokens, window_size=2):
        self.data = []
        for mid_index in range(window_size, len(encoded_tokens)-window_size):
            centre = encoded_tokens[mid_index]
            for w in range(-window_size, window_size+1):
                if w==0:
                    continue
                context = encoded_tokens[mid_index + w]
                self.data.append((centre, context))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        centre, context = self.data[index]
        return torch.tensor(centre, dtype=torch.long), torch.tensor(context, dtype=torch.long)
    

def get_dataset_and_vocab(vocab_size=60000, apply_subsampling=True, window_size=2):
    dataset = load_dataset("afmck/text8")
    
    tokens = dataset['train'][0]['text'].split()
    
    ## TO SPEED THINGS UP
    # tokens = tokens[:100_000]  # Take first 100k words only

    # Build vocabulary
    word_freq = Counter(tokens) # Counter returns a dictionary-like object mapping the tokens->freq they appear

    most_common = word_freq.most_common(vocab_size-1) # reserve 1 index for unknown value
    # most_common(n) returns the top n most frequency words and their counts as a list of tuples in descending order

    word_to_index = {}
    for index, (word, _) in enumerate(most_common):
        word_to_index[word] = index+1
    # builds the word->index dictionary starting from index 1

    word_to_index["<UNK>"] = 0 # map an unknown value to this token

    index_to_word = {}
    for word, index in word_to_index.items():
        index_to_word[index] = word

    # Subsampling Frequency
    if apply_subsampling:
        tokens = subsample(tokens, word_freq)
    
    # Encode: converting the filtered word tokens into numerical indices
    encoded_tokens = [word_to_index.get(word, 0) for word in tokens]

    # Create dataset
    dataset = SkipGramDataset(encoded_tokens, window_size=window_size)
    return dataset, word_to_index, index_to_word


if __name__ == "__main__":
    dataset, word_to_idx, idx_to_word = get_dataset_and_vocab()
    print("Sample:", dataset[0])
    print("Vocab size:", len(word_to_idx))