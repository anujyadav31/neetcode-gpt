import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        #pass
        combined = positive + negative

        # Build vocabulary: sorted unique words -> integer IDs starting at 1
        vocabulary = sorted({word for sentence in combined for word in sentence.split()})
        word_to_id = {word: idx + 1 for idx, word in enumerate(vocabulary)}

        # Encode each sentence as a tensor of word IDs
        encoded = [torch.tensor([word_to_id[w] for w in s.split()]) for s in combined]
        #print(encoded)
        # Pad shorter sequences with 0s so output is a rectangular tensor
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)       
        '''
        #print(positive)
        #print(negative)
        #print(positive + negative)
        combined = positive + negative
        #print(combined[0].split())
        temp_voc = []
        vocabulary = []
        for sentence in combined:
            temp_voc.append(sentence.split())
        #print(temp_voc)
        #print(combined)
        for i in temp_voc:
            for k in range(len(i)):
                vocabulary.append(i[k])
        #print(vocabulary)
        vocabulary = list(sorted(set(vocabulary)))
        #print(vocabulary)
        word_to_id = {word: idx+1 for idx, word in enumerate(vocabulary)}
        #print(word_to_id)
        for i in range(len(combined)):
            combined[i] = combined[i].split()
        #print(combined)
        for i in range(len(combined)):
            for k in range(len(combined[i])):
                combined[i][k] = word_to_id[combined[i][k]]
        #print(combined)
        max_len = 0
        for i in range(len(combined)):
            max_len = max(max_len, len(combined[i]))
        #print(max_len)
        for i in range(len(combined)):
            if len(combined[i]) < max_len:
                while(len(combined[i]) < max_len):
                    combined[i].append(0)
        #print(combined)
        for i in range(len(combined)):
            for k in range(len(combined[i])):
                combined[i][k]=float(combined[i][k])
        #print(combined)
        return combined'''
