import pandas as pd
import json
import os
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nltk

print("Downloading NLTK data for synonyms...")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

DATA_DIR = "data/raw/scicite/scicite"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
OUT_PATH = os.path.join(DATA_DIR, "train_augmented.jsonl")

def main():
    print(f"Loading {TRAIN_PATH}...")
    df = pd.read_json(TRAIN_PATH, lines=True)

    key_df = df[df['isKeyCitation'] == True]
    not_key_df = df[df['isKeyCitation'] == False]

    print(f"Original Dataset: {len(key_df)} Key Citations | {len(not_key_df)} Not Key Citations")

    # Initialize Synonym Augporter (replaces roughly 10% of words with synonyms)
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)

    augmented_data = []

    print("Augmenting Key Citations (Generating synthetic examples)...")
    for _, row in tqdm(key_df.iterrows(), total=len(key_df)):
        try:
            original_text = row['string']
            augmented_text = aug.augment(original_text)
            
            # Extract string from list if nlpaug returns one
            if isinstance(augmented_text, list):
                augmented_text = augmented_text[0]
                
            new_row = row.copy()
            new_row['string'] = augmented_text
            new_row['id'] = str(row['id']) + "_aug"
            augmented_data.append(new_row)
        except Exception as e:
            continue

    df_aug_only = pd.DataFrame(augmented_data)
    df_final = pd.concat([df, df_aug_only], ignore_index=True)

    print(f"Augmented Dataset Total: {len(df_final[df_final['isKeyCitation'] == True])} Key Citations | {len(not_key_df)} Not Key Citations")

    print(f"Saving to {OUT_PATH}...")
    df_final.to_json(OUT_PATH, orient='records', lines=True)
    print("Augmentation complete!")

if __name__ == '__main__':
    main()
