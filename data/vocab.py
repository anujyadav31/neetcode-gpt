from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = sorted(set(text))
        stoi = {ch : i for i, ch in enumerate(chars)}
        #print(f"stoi = {stoi}")
        #print(f"stoi.items() = {stoi.items()}")
        itos = {i : ch for ch, i in stoi.items()}
        #print(f"itos = {itos}")
        return stoi, itos
        #pass


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        self.build_vocab(text)
        #pass
        #a=[stoi[ch] for ch in text]
        #print(f"a = {a}")
        #pass
        return [stoi[ch] for ch in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        #b = [itos[i] for i in ids]
        #print(f"b = {b}")
        #c = ''.join(b)
        #print(f"c = {c}")
        #pass
        return ''.join([itos[i] for i in ids])
