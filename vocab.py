import os
import pandas as pd
from nltk.tokenize import word_tokenize
from constants import *
from torch import Tensor
import math

class Vocabulary():
    def __init__(self, root_dir, captions_file_name):
        self.caps_file = captions_file_name
        self.root_dir = root_dir
        self.word2int = {}
        self.int2word = {}
        self.words = []
        self.max_length = -1
        self.build_vocab()
        
    def build_vocab(self):
         print("Building Vocab...")
         caps_data = pd.read_csv(os.path.join(self.caps_file))

         tokens = []

         for _, cap_data in caps_data.iterrows():
             tokens.extend(self.tokenize(cap_data))
        
         self.words = list(set(tokens))
         
         self.words.extend([START_WORD, END_WORD, UNK_WORD])

         for i, word in enumerate(self.words):
             self.encode_word(word, i)
         print(self.int2word)

    def encode_word(self, word, num):
        if num not in self.int2word:
            self.word2int[word] = num
            self.int2word[num] = word
     
    def tokenize(self, caption_ann):
        tokens = []
        if caption_ann['img_id'] == float('-inf'):
            print(caption_ann['img_id'])
        for i in range(MAX_CAPTIONS):
            caption = caption_ann['caption' + str(i)]
            if isinstance(caption, str):
                self.max_length = max(self.max_length, len(caption))
                tokens.extend(word_tokenize(caption))
        return tokens
    
    def convert_to_tensor(self, caption):
        tokens = []
        
        caption = word_tokenize(caption)
        
        tokens = [self.word2int[START_WORD]]
        for _, token in enumerate(caption):
            token = (self.word2int[token] if self.word2int[token] else self.word2int[UNK_WORD])
            tokens.append(token)
        for _ in range(len(tokens), self.max_length + 2):
            tokens.append(self.word2int[END_WORD])
        tokens = Tensor(tokens).long()

        return tokens

    def get_vocab_size(self):
        return len(self.words)
    
    def get_caption(self, int_list):
        
        caption = []
        for _, id in enumerate(int_list):
            if id in self.int2word:
                word = self.int2word[id] 
                if word == END_WORD:
                    break
                if word != START_WORD:
                    caption.append(word)
            else:
                caption.append(UNK_WORD)
        
        return caption

