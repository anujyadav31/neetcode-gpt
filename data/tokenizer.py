from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        #pass
        #print(corpus)
        tokens = list(corpus)
        #print(tokens)
        merges = []
        for _ in range(num_merges):
            #print("................................")
            if len(tokens) < 2:
                break
            # Count adjacent pair frequencies
            pairs = {}
            #print(f"pairs = {pairs}")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                #print(f"pair = {pair}")
                pairs[pair] = pairs.get(pair, 0) + 1
                #print(f"pairs = {pairs}")

            if not pairs:
                break

            # Find most frequent pair (tiebreak: lexicographically smallest)
            best_count = max(pairs.values())
            #print(f"best_count = {best_count}")
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            #print(f"candidates = {candidates}")
            best = candidates[0]
            #print(f"best = {best}")

            #print(f"merges = {merges}")
            merges.append([best[0], best[1]])
            #print(f"merges = {merges}")

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            #print(f"new_tokens = {new_tokens}")
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    #print(f"new_tokens = {new_tokens}")
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    #print(f"new_tokens = {new_tokens}")
                    i += 1
                #print(f"i = {i}")
            tokens = new_tokens

        return merges

