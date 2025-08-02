import random

def subsample(tokens, word_freq, threshold=1e-5):
    
    # sum of frequencies of all words in the vocabulary
    total_count = sum(word_freq[word] for word in word_freq)
    def keep(word):
        freq = word_freq[word] / total_count
        prob = (freq / threshold)**0.5 + 1
        prob = threshold / freq * prob
        return random.random() < prob
    
    return [word for word in tokens if keep(word)]