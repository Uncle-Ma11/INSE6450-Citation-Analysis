import os
import pandas as pd
import json
from datasets import load_from_disk
import glob

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def analyze_act2():
    print("-" * 50)
    print("Analyzing ACT2...")
    act2_dir = os.path.join(RAW_DIR, "ACT2")
    if not os.path.exists(act2_dir):
        print("ACT2 directory not found.")
        return

    # Look for data files
    files = glob.glob(os.path.join(act2_dir, "**", "*.csv"), recursive=True)
    files += glob.glob(os.path.join(act2_dir, "**", "*.json"), recursive=True)
    files += glob.glob(os.path.join(act2_dir, "**", "*.tsv"), recursive=True)
    
    if not files:
        print("No data files found in ACT2 directory.")
        return

    print(f"Found {len(files)} files.")
    for f in files:
        print(f"\nChecking {os.path.basename(f)}...")
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(f)
            elif f.endswith('.tsv'):
                df = pd.read_csv(f, sep='\t')
            else:
                df = pd.read_json(f)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for interesting columns
            if 'citation_class' in df.columns:
                print("Label Distribution (citation_class):")
                print(df['citation_class'].value_counts())
            if 'citation_influence' in df.columns:
                print("Label Distribution (citation_influence):")
                print(df['citation_influence'].value_counts())
                
        except Exception as e:
            print(f"Could not read {f}: {e}")

def analyze_scicite():
    print("-" * 50)
    print("Analyzing SciCite...")
    scicite_path = os.path.join(RAW_DIR, "scicite")
    
    if not os.path.exists(scicite_path):
        print("SciCite path not found.")
        return

    # Look for jsonl files
    files = glob.glob(os.path.join(scicite_path, "**", "*.jsonl"), recursive=True)
    if not files:
         print("No JSONL files found in SciCite.")
         return
         
    for f in files:
        print(f"\nChecking {os.path.basename(f)}...")
        try:
             # Just read first few lines to infer schema or read fully if small
             # SciCite files can be large, so read line by line or use pandas with lines=True
             df = pd.read_json(f, lines=True)
             print(f"Shape: {df.shape}")
             print(f"Columns: {list(df.columns)}")
             
             if 'label' in df.columns:
                 print("Label Distribution:")
                 print(df['label'].value_counts())
             elif 'intent' in df.columns:
                 print("Intent Distribution:")
                 print(df['intent'].value_counts())

        except Exception as e:
            print(f"Failed to load {f}: {e}")

def analyze_acl_arc():
    print("-" * 50)
    print("Analyzing ACL-ARC...")
    acl_dir = os.path.join(RAW_DIR, "acl_arc")
    if not os.path.exists(acl_dir):
         print("ACL-ARC directory not found.")
         return
         
    files = glob.glob(os.path.join(acl_dir, "**", "*.csv"), recursive=True)
    if not files:
        print("No CSV files found in ACL-ARC.")
        return

    for f in files:
        print(f"\nChecking {os.path.basename(f)}...")
        try:
             df = pd.read_csv(f)
             print(f"Shape: {df.shape}")
             # Check distinct labels
             if 'intent' in df.columns:
                 print("Intent Distribution:")
                 print(df['intent'].value_counts())
        except Exception as e:
            print(f"Failed to read {f}: {e}")

def analyze_valenzuela():
    print("-" * 50)
    print("Analyzing Valenzuela...")
    val_dir = os.path.join(RAW_DIR, "valenzuela")
    if not os.path.exists(val_dir):
         print("Valenzuela directory not found.")
         return
         
    files = glob.glob(os.path.join(val_dir, "**", "*.csv"), recursive=True)
    
    if not files:
        print("No CSV files found in Valenzuela directory.")
        # Check for other text files
        files = glob.glob(os.path.join(val_dir, "**", "*.txt"), recursive=True)
    
    for f in files:
        print(f"\nChecking {os.path.basename(f)}...")
        try:
             df = pd.read_csv(f)
             print(f"Shape: {df.shape}")
             print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Could not read {f}: {e}")

def main():
    analyze_act2()
    analyze_scicite()
    analyze_acl_arc()
    analyze_valenzuela()

if __name__ == "__main__":
    main()
