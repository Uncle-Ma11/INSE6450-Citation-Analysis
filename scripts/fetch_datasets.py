import os
import requests
import zipfile
import shutil
import subprocess
from datasets import load_dataset
import sys

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fetch_act2():
    print("Fetching ACT2 dataset...")
    target_dir = os.path.join(RAW_DIR, "ACT2")
    if os.path.exists(target_dir):
        print(f"ACT2 directory already exists at {target_dir}. Skipping clone.")
        return

    repo_url = "https://github.com/oacore/ACT2.git"
    try:
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
        print("Successfully cloned ACT2.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone ACT2: {e}")

def fetch_scicite():
    print("\nFetching SciCite dataset...")
    try:
        # Try direct download from S3
        url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
        target_dir = os.path.join(RAW_DIR, "scicite")
        ensure_dir(target_dir)
        
        tar_path = os.path.join(target_dir, "scicite.tar.gz")
        
        if os.path.exists(tar_path):
             print(f"SciCite tar already exists at {tar_path}")
        else:
             print(f"Downloading SciCite from {url}...")
             response = requests.get(url, stream=True)
             if response.status_code == 200:
                 with open(tar_path, 'wb') as f:
                     for chunk in response.iter_content(chunk_size=8192):
                         f.write(chunk)
                 print("Successfully downloaded SciCite.")
             else:
                 print(f"Failed to download SciCite (Status: {response.status_code})")
                 return

        # Extract
        import tarfile
        if tarfile.is_tarfile(tar_path):
            print("Extracting SciCite...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=target_dir)
            print("Successfully extracted SciCite.")
        
    except Exception as e:
        print(f"Failed to fetch SciCite: {e}")

def fetch_acl_arc():
    print("\nFetching ACL-ARC (citation_intent)...")
    try:
        # Load citation_intent from Hugging Face
        # Search suggests 'hrithikpiyush/acl-arc' is a candidate or 'citation_intent'
        try:
             dataset = load_dataset("hrithikpiyush/acl-arc")
        except:
             print("Failed to load 'hrithikpiyush/acl-arc', trying 'citation_intent' again...")
             dataset = load_dataset("citation_intent")
        
        target_dir = os.path.join(RAW_DIR, "acl_arc")
        ensure_dir(target_dir)
        
        # Save as CSV for consistency with project
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            # Map labels if needed, or just save raw
            # The processing logic from deprecated script mapped labels.
            # We will save raw here and let analysis/processing script handle mapping?
            # Or map here? The user liked 'acl_arc_train.csv' which had mapping.
            # Let's save raw first to be a "fetcher".
            save_path = os.path.join(target_dir, f"{split}.csv")
            # escaping is needed for some datasets with text fields
            df.to_csv(save_path, index=False, escapechar='\\')
            print(f"Saved {split} to {save_path}")
            
    except Exception as e:
        print(f"Failed to fetch ACL-ARC: {e}")

def fetch_valenzuela():
    print("\nFetching Valenzuela (Meaningful Citations) dataset...")
    target_dir = os.path.join(RAW_DIR, "valenzuela")
    ensure_dir(target_dir)
    
    # URL guesses based on AI2 public datasets
    # "https://ai2-public-datasets.s3.amazonaws.com/meaningful-citations/citation_data.zip" is a common guess
    # Or "https://s3-us-west-2.amazonaws.com/ai2-s2-research/legacy/meaningful_citations/meaningful_citations.zip"
    
    possible_urls = [
        "https://ai2-public-datasets.s3.amazonaws.com/meaningful-citations/citation_data.zip",
        "https://ai2-public-datasets.s3.amazonaws.com/meaningful-citations/dataset.zip",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-research/legacy/meaningful_citations/meaningful_citations.zip"
    ]
    
    file_name = "dataset.zip"
    save_path = os.path.join(target_dir, file_name)
    
    if os.path.exists(save_path):
         print(f"File already exists at {save_path}. Skipping download.")
         return

    success = False
    for url in possible_urls:
        try:
            print(f"Trying {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded from {url}")
                success = True
                break
            else:
                print(f"Failed with status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
            
    if success:
        try:
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Successfully extracted dataset.")
        except zipfile.BadZipFile:
            print("Downloaded file is not a valid zip file.")
    else:
        print("Could not download Valenzuela dataset from known URLs.")
        print("Please download manually from https://allenai.org/data/meaningful-citations and place in data/raw/valenzuela/")

def main():
    ensure_dir(RAW_DIR)
    
    fetch_act2()
    fetch_scicite()
    fetch_acl_arc()
    fetch_valenzuela()
    
    print("\nFetching complete.")

if __name__ == "__main__":
    main()
