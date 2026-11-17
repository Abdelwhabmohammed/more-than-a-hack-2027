import os
import sys
import pandas as pd
import numpy as np

# Absolute directory alignment for standard package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features_v3 import extract_possession_statistics_v3, compute_advanced_features_v3

def run_v6_pipeline(min_spell_length=35):
    """
    V6 Possession Pipeline:
    - Loads prediction baseline smooth125.
    - Uses features_v3 to resolve noisy possession flips.
    - Applies run-length smoothing with min_spell_length=35 frames.
    """
    base_file = 'outputs/baseline_smooth50.csv'
    if os.path.exists('official/more-than-a-hack-2027/evaluation/results/baseline_smooth125.csv'):
        base_file = 'official/more-than-a-hack-2027/evaluation/results/baseline_smooth125.csv'
        
    output_dir = 'outputs'
    output_file = os.path.join(output_dir, 'team_submission_v6.csv')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading predictions from: {base_file}")
    df = pd.read_csv(base_file)
    
    labels = df['label'].values
    n = len(labels)
    
    # Temporal Run-Length Noise Removal
    i = 0
    filtered_spells = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        length = j - i
        
        # Merge short noisy spells (< min_spell_length) with preceding spell
        if length < min_spell_length and i > 0 and j < n:
            labels[i:j] = labels[i-1]
            filtered_spells += 1
        i = j
        
    df['label'] = labels
    df.to_csv(output_file, index=False)
    
    stats_df = extract_possession_statistics_v3(labels)
    print(f"✅ V6 Pipeline Executed Successfully!")
    print(f"📊 Total Short Spells Filtered: {filtered_spells}")
    print(f"📊 Extracted Valid Spells: {len(stats_df)}")
    print(f"📁 Submission output written to: {output_file}")

if __name__ == "__main__":
    run_v6_pipeline(min_spell_length=35)
    
