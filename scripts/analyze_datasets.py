import os
import pandas as pd
import json
import glob

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def print_dataset_details(name, df, label_col=None):
    print(f"\n{'='*20} {name} Analysis {'='*20}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n--- Feature Explanations (Inferred) ---")
    for col in df.columns:
        # Get a non-null example if possible
        non_null_series = df[col].dropna()
        if not non_null_series.empty:
            example = non_null_series.iloc[0]
        else:
            example = "All Null"
            
        print(f"- {col}: Type {df[col].dtype}, Example: {str(example)[:100]}...")

    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")

    if label_col and label_col in df.columns:
        print(f"\n--- Label Distribution ({label_col}) ---")
        counts = df[label_col].value_counts()
        percentages = df[label_col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        print(dist_df)
    
    print("="*60)

def analyze_act2():
    act2_dir = os.path.join(RAW_DIR, "ACT2")
    if not os.path.exists(act2_dir):
        print("ACT2 directory not found.")
        return

    files = glob.glob(os.path.join(act2_dir, "**", "*.csv"), recursive=True)
    files += glob.glob(os.path.join(act2_dir, "**", "*.tsv"), recursive=True)
    
    if not files:
        print("No files found in ACT2.")
        return

    for f in files:
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, sep='\t')
            
            label_col = None
            if 'citation_class' in df.columns:
                label_col = 'citation_class'
            elif 'citation_influence' in df.columns:
                label_col = 'citation_influence'
                
            print_dataset_details(f"ACT2 - {os.path.basename(f)}", df, label_col)
                
        except Exception as e:
            print(f"Could not read {f}: {e}")

def analyze_scicite():
    scicite_path = os.path.join(RAW_DIR, "scicite")
    if not os.path.exists(scicite_path):
        print("SciCite path not found.")
        return

    files = glob.glob(os.path.join(scicite_path, "**", "*.jsonl"), recursive=True)
    if not files:
        print("No files found in SciCite.")
        return

    for f in files:
        try:
            # SciCite files can be large, reading entire file for analysis
            df = pd.read_json(f, lines=True)
            label_col = 'label' if 'label' in df.columns else None
            print_dataset_details(f"SciCite - {os.path.basename(f)}", df, label_col)

        except Exception as e:
            print(f"Failed to load {f}: {e}")

def analyze_acl_arc():
    acl_dir = os.path.join(RAW_DIR, "acl_arc")
    if not os.path.exists(acl_dir):
         print("ACL-ARC directory not found.")
         return
         
    files = glob.glob(os.path.join(acl_dir, "**", "*.csv"), recursive=True)
    if not files:
        print("No files found in ACL-ARC.")
        return

    for f in files:
        try:
             df = pd.read_csv(f)
             label_col = 'intent' if 'intent' in df.columns else None
             print_dataset_details(f"ACL-ARC - {os.path.basename(f)}", df, label_col)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

def main():
    print("Starting Dataset Analysis...")
    analyze_act2()
    analyze_scicite()
    analyze_acl_arc()
    # analyze_valenzuela() # Excluded as per request
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
