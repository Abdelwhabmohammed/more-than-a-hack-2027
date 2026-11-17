import os
import pandas as pd
import numpy as np
from features_v2 import extract_possession_statistics

def run_v5_pipeline(min_spell_length=30):
    # Try reading baseline smooth125 from outputs or evaluation directory
    base_file = 'outputs/baseline_smooth50.csv'
    if os.path.exists('official/more-than-a-hack-2027/evaluation/results/baseline_smooth125.csv'):
        base_file = 'official/more-than-a-hack-2027/evaluation/results/baseline_smooth125.csv'
        
    output_dir = 'outputs'
    output_file = os.path.join(output_dir, 'team_submission_v5.csv')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading predictions from: {base_file}")
    df = pd.read_csv(base_file)
    
    labels = df['label'].values
    n = len(labels)
    
    # Apply Short-Spell Noise Removal (< 30 frames)
    i = 0
    filtered_count = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        length = j - i
        
        if length < min_spell_length and i > 0 and j < n:
            labels[i:j] = labels[i-1]
            filtered_count += 1
        i = j
        
    df['label'] = labels
    df.to_csv(output_file, index=False)
    
    # Calculate stats using features_v2
    stats_df = extract_possession_statistics(labels)
    print(f"✅ V5 Pipeline Done! Total Spells extracted: {len(stats_df)}")
    print(f"📁 Submission saved to: {output_file}")

if __name__ == "__main__":
    run_v5_pipeline(min_spell_length=30)
