import numpy as np
from scipy.signal import argrelextrema, find_peaks
import matplotlib.pyplot as plt
import json
from options.base_options import BaseOptions

def get_continuous_indices(sii_values, N, total_channels):
    sii_values = np.array(sii_values)
    extrema_indices = argrelextrema(sii_values, np.greater, order=1)[0]
    extrema_values = sii_values[extrema_indices]
    sorted_indices = np.argsort(extrema_values)[::-1]
    sorted_extrema = [(extrema_indices[i], extrema_values[i]) for i in sorted_indices]
    
    e1 = sorted_extrema[0]
    e2 = sorted_extrema[1]
    dist = abs(e1[0] - e2[0])
    
    if dist >= N:
        if e1[0] < e2[0]:
            start_idx = e1[0]
            end_idx = start_idx + N
        else:
            end_idx = e1[0] + 1
            start_idx = end_idx - N
            

        start_idx = max(0, start_idx)
        end_idx = min(total_channels, end_idx)
        return list(range(start_idx, end_idx))

    L = min(e1[0], e2[0])
    R = max(e1[0], e2[0])
    
    for i in range(2, len(sorted_extrema)):
        curr_e = sorted_extrema[i]
        curr_idx = curr_e[0]
        
        if curr_idx < L:
            L = curr_idx
        elif curr_idx > R:
            R = curr_idx
            

        current_len = R - L + 1 
        if current_len >= N:
            if e1[0] <= e2[0]: 
                return list(range(L, L + N))
            else:
                return list(range(R - N + 1, R + 1))

    
    current_len = R - L + 1
    needed = N - current_len
    expand_left = 0
    expand_right = 0
    
    if e1[0] <= e2[0]:
        can_expand_left = L 
        
        take_left = min(can_expand_left, needed)
        L -= take_left
        needed -= take_left
        
        if needed > 0:
            can_expand_right = total_channels - 1 - R
            take_right = min(can_expand_right, needed)
            R += take_right
            needed -= take_right
    else:
        can_expand_right = total_channels - 1 - R
        take_right = min(can_expand_right, needed)
        R += take_right
        needed -= take_right
        
        if needed > 0:
            can_expand_left = L
            take_left = min(can_expand_left, needed)
            L -= take_left
            needed -= take_left
            
    return list(range(L, R + 1))

def select_continuous_indices(mean_amplitudes, target_count, min_distance=50):
    total_layers = len(mean_amplitudes)
    if target_count >= total_layers:
        return list(range(total_layers))
    peaks, _ = find_peaks(mean_amplitudes, distance=min_distance)
    
    if len(peaks) == 0:
        seeds = [np.argmax(mean_amplitudes)]
    else:
        sorted_peaks = sorted(peaks, key=lambda x: mean_amplitudes[x], reverse=True)
        seeds = sorted_peaks[:2]


    selected_indices = set(seeds)
    candidates = set()
    
    def add_neighbors(idx):
        for neighbor in [idx - 1, idx + 1]:
            if 0 <= neighbor < total_layers and neighbor not in selected_indices:
                candidates.add(neighbor)

    for seed in seeds:
        add_neighbors(seed)

    while len(selected_indices) < target_count and candidates:
        best_candidate = max(candidates, key=lambda x: mean_amplitudes[x])
        selected_indices.add(best_candidate)
        candidates.remove(best_candidate)
        add_neighbors(best_candidate)

    if len(selected_indices) < target_count:
        remaining = list(set(range(total_layers)) - selected_indices)
        remaining.sort(key=lambda x: mean_amplitudes[x], reverse=True)
        selected_indices.update(remaining[:target_count - len(selected_indices)])

    return sorted(list(selected_indices))


if __name__ == "__main__":
    N = BaseOptions().parse().slicelen
    with open('.../SII_6mm.json', 'r') as f:
        data = json.load(f)
    select_index = {}
    for sample in data:
        name = sample['name']
        SII = sample['SII']
        index = []
        continuous_list = select_continuous_indices(SII, N)
        L = int(continuous_list[0])
        R = int(continuous_list[-1]+1)
        index.append(L)
        index.append(R)
        print(f'the name is {name}, the list is {L} and {R}')
        select_index[name] = index
    with open('.../6mm_N.json', 'w') as f:
        json.dump(select_index, f, indent=4)