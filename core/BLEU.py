from collections import Counter
from nltk import ngrams
import numpy as np

def n_gram(candidate, ref_arr, n):
    cand = Counter(ngrams(candidate, n))

    result_arr = []
    for ref in ref_arr:
        result = Counter(ngrams(ref, n))
        result_arr.append(result)
        
    score = 0 
    for key, value in cand.items():
        ref_c = 0
        for result in result_arr:
            ref_c = max(ref_c, result.get(key, 0))
        score += min(ref_c, value)

    return score / (sum(cand.values()) + 1e-9)

def closet_ref_length(candidate, ref_arr):
    cand_len = len(candidate)
    ref_lens = (len(ref) for ref in ref_arr)
    closet_ref_len = min(ref_lens, key=lambda ref_len : (abs(ref_len - cand_len), ref_len))
    return closet_ref_len

def brevity_penalty(candidate, ref_arr):
    cand_len = len(candidate)
    ref_len = closet_ref_length(candidate, ref_arr)

    if cand_len > ref_len:
        return 1
    elif cand_len == 0:
        return 0
    else:
        return np.exp(1 - ref_len/cand_len)

def BLEU(candidate, ref_arr, total_n):
    score = 0
    
    # mean of n-gram Precision 
    weight = 1 / total_n
    # for n in range(total_n):
    #     print(n_gram(candidate, ref_arr, n + 1))
    # BLEU 
    score_arr = [weight * np.log(n_gram(candidate, ref_arr, n + 1) + 1e-5) for n in range(total_n)]
    score = np.exp(np.sum(score_arr))

    # brevity_penalty
    b_p = brevity_penalty(candidate, ref_arr)
    
    score *= b_p
    return score