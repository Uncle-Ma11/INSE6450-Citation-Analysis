
from datasets import load_dataset
import pandas as pd

def map_acl_to_project(example):
    mapping = {
        0: 'Perfunctory',    # Background -> Perfunctory (approx)
        1: 'Operational',    # Uses -> Operational
        2: 'Juxtapositional',# CompareOrContrast -> Juxtapositional
        3: 'Evolutionary',   # Extends -> Evolutionary
        4: 'Conceptual',     # Motivation -> Conceptual
        5: 'Other'           # Future -> Other
    }
    # citation_intent has 'intent' as the label column
    example['project_label'] = mapping[example['intent']]
    return example

def main():
    print("Loading ACL-ARC (citation_intent) dataset...")
    # The dataset is often named 'citation_intent' in huggingface datasets
    # Trying alternative names if default fails
    dataset_names = ["zapsdcn/citation_intent", "citation_intent", "allenai/scicite"]
    
    dataset = None
    loaded_name = None
    for name in dataset_names:
        try:
            print(f"Trying to load {name}...")
            # trust_remote_code might be needed for some, but discouraged. 
            # If it fails, we might need to remove it or set it to False.
            dataset = load_dataset(name, trust_remote_code=True)
            print(f"Successfully loaded {name}")
            loaded_name = name
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            try:
                # Retry without trust_remote_code if that was the issue? 
                # Or just move on. 
                pass
            except:
                pass
    
    if dataset is None:
        print("Could not load any dataset.")
        return

    print(f"\nFinal Loaded Dataset: {loaded_name}")
    print("\nDataset Structure:")
    print(dataset)
    
    # Check for label column
    label_col = 'intent'
    if 'intent' not in dataset['train'].column_names:
        if 'label' in dataset['train'].column_names:
            label_col = 'label'
        elif 'section_label' in dataset['train'].column_names:
             label_col = 'section_label'
    
    print(f"\nIdentified label column: {label_col}")
    if label_col in dataset['train'].features:
         print(f"Features for {label_col}: {dataset['train'].features[label_col]}")
         if hasattr(dataset['train'].features[label_col], 'names'):
             print(f"Label Names: {dataset['train'].features[label_col].names}")
    else:
         print(f"Column {label_col} not found in features.")

    # Inspect unique values in label column
    unique_labels = sorted(set(dataset['train'][label_col]))
    print(f"\nUnique values in '{label_col}': {unique_labels}")
    print(f"Type of first label: {type(unique_labels[0])}")

    # Update mapping function to use the correct label column
    def map_acl_to_project_dynamic(example):
        val = example[label_col]
        
        # Define mapping for standard 6-class intent
        # Map string labels if necessary
        mapping = {
            # Int keys
            0: 'Perfunctory',    
            1: 'Operational',    
            2: 'Juxtapositional',
            3: 'Evolutionary',   
            4: 'Conceptual',     
            5: 'Other',
            # String keys (case-insensitive handling might be needed, but starting with exact)
            'Background': 'Perfunctory',
            'Uses': 'Operational',
            'CompareOrContrast': 'Juxtapositional',
            'Extends': 'Evolutionary',
            'Motivation': 'Conceptual',
            'Future': 'Other'
        }
        
        example['project_label'] = mapping.get(val, 'Unknown')
        return example

    processed_data = dataset.map(map_acl_to_project_dynamic)
    
    # Inspect one value to debug
    first_val = processed_data['train'][0][label_col]
    print(f"\nDebug: First example label value: {first_val} (Type: {type(first_val)})")
    print(f"Debug: First example project label: {processed_data['train'][0]['project_label']}")
    
    # Convert to pandas for easier analysis
    train_df = processed_data['train'].to_pandas()
    
    # Save to CSV for the user
    output_path = 'data/acl_arc_train.csv'
    train_df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to: {output_path}")
    
    if label_col in dataset['train'].features and hasattr(dataset['train'].features[label_col], 'names'):
        names = dataset['train'].features[label_col].names
        train_df['original_label_name'] = train_df[label_col].apply(lambda x: names[x] if x < len(names) else str(x))
    else:
        train_df['original_label_name'] = train_df[label_col]

    print("\nOriginal Label Distribution (Train set):")
    print(train_df['original_label_name'].value_counts())

    print("\nProject Label Distribution (Train set):")
    print(train_df['project_label'].value_counts())

    print("\nExamples:")
    # Sample 5 random examples
    if len(train_df) > 5:
        sample = train_df.sample(5)
    else:
        sample = train_df
        
    for idx, row in sample.iterrows():
        print(f"-" * 80)
        text = row['text'] if 'text' in row else row.get('string', str(row))
        print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
        print(f"Original: {row['original_label_name']} | Mapped: {row['project_label']}")

if __name__ == "__main__":
    main()
