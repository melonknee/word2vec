import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, centre_words):
        # centre_words: (batch_size,)
        embed = self.input_embeddings(centre_words) # (batch_size, embedding_dim)
        return embed
    
    def get_output_scores(self, embed):
        # embed: (batch_size, embedding_dim)
        # Return scores for all context words
        return torch.matmul(embed, self.output_embeddings.weight.T)
    
    def get_embeddings(self):
        return self.input_embeddings.weight.data
    

if __name__ == "__main__":
    model = SkipGramModel(vocab_size=60000, embedding_dim=100)
    dummy_input = torch.tensor([1, 2, 3])
    embed = model(dummy_input)
    print("Embed shape:", embed.shape)
    scores = model.get_output_scores(embed)
    print("Scores shape:", scores.shape)  # (batch_size, vocab_size)