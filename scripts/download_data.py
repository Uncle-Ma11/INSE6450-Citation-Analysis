import os
import requests
import tarfile
import shutil

DATA_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
RAW_DATA_DIR = os.path.join("data", "raw")
TAR_FILE_PATH = os.path.join(RAW_DATA_DIR, "scicite.tar.gz")

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        exit(1)

def extract_tar(tar_path, target_dir):
    print(f"Extracting {tar_path} to {target_dir}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        exit(1)

def main():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    
    if not os.path.exists(TAR_FILE_PATH):
        download_file(DATA_URL, TAR_FILE_PATH)
    else:
        print(f"File {TAR_FILE_PATH} already exists. Skipping download.")
        
    extract_tar(TAR_FILE_PATH, RAW_DATA_DIR)
    
    # Organize if needed (the tar usually extracts into a subdir)
    # Check contents
    print(f"Contents of {RAW_DATA_DIR}:")
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for name in files:
            print(os.path.join(root, name))

if __name__ == "__main__":
    main()
