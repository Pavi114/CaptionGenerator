import numpy as np

def embed_captions():
    embedding_index = {}

    captions = open('./glove.6B.100d.txt', encoding="utf-8")

    for caption in captions:
        values = caption.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs
    
    return embedding_index