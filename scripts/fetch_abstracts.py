
import json
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import quote

# Configuration
DATA_DIR = "data/raw/scicite/scicite"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
DEV_PATH = os.path.join(DATA_DIR, "dev.jsonl")
OUTPUT_PATH = os.path.join(DATA_DIR, "abstracts_mapping.json")

# API Configs
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_API_URL = "https://api.openalex.org/works"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Politeness
EMAIL = os.environ.get("CROSSREF_EMAIL", "researcher@example.com") 
HEADERS = {
    "User-Agent": f"CitationAnalysisBot/1.0 ({EMAIL})"
}

BATCH_SIZE_S2 = 50 
BATCH_SIZE_OPENALEX = 50 # OpenAlex filter limit might be 50-100
BATCH_SIZE_ARXIV = 50

S2_FIELDS = "paperId,title,abstract,externalIds,url"

def get_unique_cited_ids(file_paths):
    unique_ids = set()
    for path in file_paths:
        if os.path.exists(path):
            print(f"Reading {path}...")
            df = pd.read_json(path, lines=True)
            if 'citedPaperId' in df.columns:
                unique_ids.update(df['citedPaperId'].dropna().unique())
    return list(unique_ids)

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    tokens = []
    for word, positions in inverted_index.items():
        for pos in positions:
            tokens.append((pos, word))
    tokens.sort(key=lambda x: x[0])
    return " ".join([t[1] for t in tokens])

def fetch_openalex_batch(dois_map):
    """
    Fetch abstracts for a batch of DOIs.
    dois_map: {doi: paper_id, ...}
    """
    if not dois_map:
        return {}
        
    results = {}
    
    # Construct filter
    # clean DOIs
    clean_dois = []
    doi_to_pid = {}
    
    for doi, pid in dois_map.items():
        clean = doi.replace("doi.org/", "").replace("http://", "").replace("https://", "")
        clean_dois.append(f"https://doi.org/{clean}")
        doi_to_pid[f"https://doi.org/{clean}"] = pid
        
    doi_filter = "|".join(clean_dois)
    url = f"{OPENALEX_API_URL}?filter=doi:{doi_filter}&per-page={len(clean_dois)}"
    
    if EMAIL:
        if "?" in url: url += f"&mailto={EMAIL}"
        else: url += f"?mailto={EMAIL}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            data = response.json()
            for work in data.get('results', []):
                doi_url = work.get('doi') # https://doi.org/10.xxxx/yyyy
                if doi_url in doi_to_pid:
                    pid = doi_to_pid[doi_url]
                    abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
                    if abstract:
                        results[pid] = abstract
        elif response.status_code == 429:
             print("OpenAlex Rate Limit (Batch). Sleeping 5s...")
             time.sleep(5)
    except Exception as e:
        print(f"OpenAlex Batch Error: {e}")
        
    return results

def fetch_arxiv_batch(arxiv_map):
    """
    Fetch abstracts for a batch of ArXiv IDs.
    arxiv_map: {arxiv_id: paper_id, ...}
    """
    if not arxiv_map:
        return {}
        
    results = {}
    
    id_list = ",".join(arxiv_map.keys())
    url = f"{ARXIV_API_URL}?id_list={id_list}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            # Parse XML manually to avoid dependencies
            content = response.text
            # Split by <entry>
            entries = content.split('<entry>')
            for entry in entries[1:]: # Skip preamble
                # Extract ID
                id_start = entry.find('<id>')
                id_end = entry.find('</id>')
                if id_start == -1: continue
                
                # ArXiv ID in XML is usually http://arxiv.org/abs/1234.5678
                full_id_url = entry[id_start+4:id_end].strip()
                # we need to match it back to our map keys. 
                # Our keys are likely Just the ID (1234.5678).
                # Let's check which key is contained in the URL
                matched_arxiv_id = None
                for aid in arxiv_map.keys():
                    if aid in full_id_url:
                        matched_arxiv_id = aid
                        break
                
                if matched_arxiv_id:
                    # Extract Summary
                    sum_start = entry.find('<summary>')
                    sum_end = entry.find('</summary>')
                    if sum_start != -1:
                        abstract = entry[sum_start+9:sum_end].strip()
                        pid = arxiv_map[matched_arxiv_id]
                        results[pid] = abstract
                        
    except Exception as e:
        print(f"ArXiv Batch Error: {e}")
        
    return results

def fetch_from_openalex_title(title):
    # Fallback for Title Search (cannot be batched easily)
    try:
        url = f"{OPENALEX_API_URL}?filter=title.search:{quote(title)}&per-page=1"
        if EMAIL: url += f"&mailto={EMAIL}"
        
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
             data = response.json()
             if data.get('results'):
                 first = data['results'][0]
                 return reconstruct_abstract(first.get('abstract_inverted_index'))
    except:
        pass
    return None

def fetch_abstracts(paper_ids):
    mapping = {}
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Loading existing mapping from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
    to_fetch = [pid for pid in paper_ids if pid not in mapping or mapping[pid] is None]
    print(f"Total IDs: {len(paper_ids)}. Need to fetch/refetch: {len(to_fetch)}.")

    if not to_fetch:
        return mapping

    # Phase 1: S2
    batches = [to_fetch[i:i + BATCH_SIZE_S2] for i in range(0, len(to_fetch), BATCH_SIZE_S2)]
    print(f"Phase 1: S2 ({len(batches)} batches)...")
    
    metadata_cache = {} # pid -> {doi, arxiv, title}

    for batch in tqdm(batches, desc="S2 Batch"):
        try:
            payload = {"ids": batch}
            response = requests.post(S2_API_URL, params={"fields": S2_FIELDS}, json=payload, headers=HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if not item or 'paperId' not in item: continue
                    pid = item['paperId']
                    
                    if item.get('abstract'):
                        mapping[pid] = item['abstract']
                    else:
                        # Cache for Phase 2
                        ext = item.get('externalIds') or {}
                        metadata_cache[pid] = {
                            'doi': ext.get('DOI'),
                            'arxiv': ext.get('ArXiv'),
                            'title': item.get('title')
                        }
                        if pid not in mapping: mapping[pid] = None
            elif response.status_code == 429:
                time.sleep(10)
            time.sleep(0.5)
            
            # Save Intermediately
            if len(mapping) % 500 == 0:
                 with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"S2 Error: {e}")
            time.sleep(1)

    # Phase 2: ArXiv Batch
    # Collect all PIDs with ArXiv IDs but no abstract
    arxiv_queue = {} # arxiv_id -> pid
    for pid, meta in metadata_cache.items():
        if mapping.get(pid) is None and meta.get('arxiv'):
            arxiv_queue[meta['arxiv']] = pid
            
    if arxiv_queue:
        print(f"Phase 2: ArXiv Batch ({len(arxiv_queue)} papers)...")
        arxiv_items = list(arxiv_queue.items())
        # Processing in chunks
        for i in tqdm(range(0, len(arxiv_items), BATCH_SIZE_ARXIV), desc="ArXiv"):
            chunk = arxiv_items[i:i+BATCH_SIZE_ARXIV]
            sub_map = {k:v for k,v in chunk}
            
            results = fetch_arxiv_batch(sub_map)
            mapping.update(results)
            time.sleep(1) # Politeness

    # Phase 3: OpenAlex DOI Batch
    # Collect all PIDs with DOI but no abstract
    doi_queue = {} # doi -> pid
    for pid, meta in metadata_cache.items():
        if mapping.get(pid) is None and meta.get('doi'):
            doi_queue[meta['doi']] = pid
            
    if doi_queue:
        print(f"Phase 3: OpenAlex DOI Batch ({len(doi_queue)} papers)...")
        doi_items = list(doi_queue.items())
        for i in tqdm(range(0, len(doi_items), BATCH_SIZE_OPENALEX), desc="OpenAlex DOI"):
            chunk = doi_items[i:i+BATCH_SIZE_OPENALEX]
            sub_map = {k:v for k,v in chunk}
            
            results = fetch_openalex_batch(sub_map)
            mapping.update(results)
            time.sleep(0.5)
            
            # Periodic Save
            if i % 500 == 0:
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Phase 4: Title Fallback (Sequential) - Only if really needed, limit to 20 to avoid long waits?
    # Or just do it. Let's do it for ALL remaining.
    title_queue = [pid for pid in metadata_cache if mapping.get(pid) is None and metadata_cache[pid].get('title')]
    
    print(f"Phase 4: OpenAlex Title Fallback ({len(title_queue)} papers)...")
    # This is slow, so maybe we skip it or limit it?
    # Let's try fetching just a few or all if count is low.
    # If count is high (>500), this will take time.
    # We will run it but catch interrupts.
    
    try:
        for pid in tqdm(title_queue, desc="Title Search"):
            title = metadata_cache[pid]['title']
            abstract = fetch_from_openalex_title(title)
            if abstract:
                mapping[pid] = abstract
            time.sleep(0.3)
            
            if len(mapping) % 50 == 0:
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("Interrupted Title Search. Saving...")

    # Final Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    total = len(mapping)
    found = sum(1 for v in mapping.values() if v is not None)
    print(f"Done. Total IDs mapped: {total}")
    print(f"Abstracts found: {found} ({found/total:.2%})")

def main():
    print("Gathering cited paper IDs...")
    ids = get_unique_cited_ids([TRAIN_PATH, DEV_PATH])
    print(f"Found {len(ids)} unique cited paper IDs.")
    fetch_abstracts(ids)

if __name__ == "__main__":
    main()
