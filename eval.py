import pickle
import torch
import torch.nn.functional as f

embeddings = torch.load("./artifacts/skipgram_embeddings.pt")
word_to_index = pickle.load(open('./artifacts/word_to_index.pkl', 'rb'))
index_to_word = pickle.load(open('./artifacts/index_to_word.pkl', 'rb'))

norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
def find_top_k_similar_words(word, top_k=5):
    if word not in word_to_index:
        print(f'{word} not found in vocabulary. Please choose another word.\n')
        return
    
    index_of_chosen_word = word_to_index[word]
    vector_of_chosen_word = embeddings[index_of_chosen_word]

    # Prepare for cosine similarity analysis
    normalised_vector = f.normalize(vector_of_chosen_word.unsqueeze(0), # shape becomes [1, embedding_dim]
                         p=2, dim=1)
    # normalise the vector to unit length using L2 norm, so it's ready for cosine similarity


    cosine_sims = torch.matmul(norm_embeddings, normalised_vector.squeeze())
    # tensor of shape [vocab_size] where each value is the similarity between "word" and every other word in the vocab

    top_val, top_index = torch.topk(cosine_sims, 6)
    # top_index contains the indices of the 6 most similar words (including the word itself)
    # top_val contains the actual cosine similarity scores

    print(f'Top {top_k} words similar to "{word}":')
    count = 0
    for i, index in enumerate(top_index):
        word = index_to_word[index.item()]
        sim = top_val[i].item()
        print(f'    {word}: {sim:4f}')
        count+=1
        if count==top_k:
            break

if __name__ == "__main__":
    print("Type a word to find its most similar words (type 'exit' to quit)\n")
    while True:
        user_input = input("Enter a word: ").strip().lower()
        if user_input == "exit":
            break
        else:
            print("")
            find_top_k_similar_words(user_input)
            print("")