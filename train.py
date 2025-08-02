import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataset_and_vocab
from model import SkipGramModel

import os
import pickle
os.makedirs("artifacts", exist_ok=True)


# ==== CONFIG ====
embedding_dim = 100
window_size = 2
batch_size = 512
epochs = 3
learning_rate = 0.003
vocab_size = 60000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checking if GPU is accessible
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")

# ~~ 1. Load dataset ~~
print("Loading dataset...")
dataset,word_to_index,index_to_word = get_dataset_and_vocab(
    vocab_size=vocab_size,
    apply_subsampling=True,
    window_size=window_size
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ~~ 2. Create Model ~~
print("Initialising Model...")
model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

# ~~ 3. Loss and Optimiser ~~
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# ~~ 4. Training loop ~~
print("Training...")
for epoch in range(epochs):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100, leave=False) #tqdm
    for centre_words, context_words in dataloader:
        centre_words = centre_words.to(device)
        context_words = context_words.to(device)

        # Forward pass
        input_embeds = model(centre_words) # (batch_size, embedding_dim)
        logits = model.get_output_scores(input_embeds) # (batch_size, vocab_size)

        # Compute loss (targets are class indices, so it's like a classification problem)
        loss = criterion(logits, context_words)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item()) #tqdm

    tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ~~ 5. Save learned embeddings ~~

print("PyTorch version when saving:", torch.__version__)
torch.save(model.state_dict(), "artifacts/skipgram_model.pt")
torch.save(model.get_embeddings().cpu(), "artifacts/skipgram_embeddings.pt")
# saves a tensor of shape (vocab_size, embedding_dim)

# Save vocab dictionaries
with open("artifacts/word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

with open("artifacts/index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)

print("✅ Model, embeddings and vocab saved successfully!")