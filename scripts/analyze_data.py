import json
import os
from collections import Counter

RAW_DATA_DIR = os.path.join("data", "raw", "scicite")
SPLITS = ["train.jsonl", "dev.jsonl", "test.jsonl"]

def analyze_split(file_path):
    print(f"\nAnalyzing {file_path}...")
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        label_counts = Counter()
        total_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    label = data.get("label", "unknown")
                    label_counts[label] += 1
                    total_count += 1
                except json.JSONDecodeError:
                    print("Error decoding JSON line")

        print(f"Total entries: {total_count}")
        print("Label Distribution:")
        for label, count in label_counts.most_common():
            percentage = (count / total_count) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
            
        return label_counts, total_count

    except Exception as e:
        print(f"Analysis failed: {e}")
        return None, 0

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory not found: {RAW_DATA_DIR}. Did you run download_data.py?")
        return

    overall_counts = Counter()
    total_all = 0

    for split in SPLITS:
        file_path = os.path.join(RAW_DATA_DIR, split)
        counts, total = analyze_split(file_path)
        if counts:
            overall_counts.update(counts)
            total_all += total

    print(f"\nOverall Dataset Stats:")
    print(f"Total entries: {total_all}")
    print("Aggregate Label Distribution:")
    for label, count in overall_counts.most_common():
        percentage = (count / total_all) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
