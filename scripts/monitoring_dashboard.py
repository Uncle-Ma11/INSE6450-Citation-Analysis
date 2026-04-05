import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    """
    def scale_range(input_array, min_val, max_val):
        input_array += -(np.min(input_array))
        input_array /= np.max(input_array) / (max_val - min_val)
        input_array += min_val
        return input_array

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
    return psi_value

def simulate_stream(df, batches=5):
    """
    Simulates a stream of data by splitting a dataframe into temporal batches.
    We introduce synthetic drift in the later batches to test the monitoring.
    """
    batch_size = len(df) // batches
    stream_data = []
    
    base_date = datetime.datetime.now()
    
    for i in range(batches):
        batch = df.iloc[i*batch_size : (i+1)*batch_size].copy()
        
        # Introduce synthetic drift in batch 3 and 4 (e.g., text texts get shorter, rare sections peak)
        if i >= 3:
            # Mask strings to make them shorter
            batch['string'] = batch['string'].apply(lambda x: x[:len(x)//2] if isinstance(x, str) else x)
            # Increase missing values
            batch.loc[batch.sample(frac=0.2).index, 'sectionName'] = None
            
        batch['timestamp'] = [base_date + datetime.timedelta(hours=i*2) for _ in range(len(batch))]
        batch['batch_id'] = i
        stream_data.append(batch)
        
    return stream_data

def generate_dashboard(stream_data, reference_df, output_path):
    # Reference distribution for feature: String length
    ref_lengths = reference_df['string'].apply(lambda x: len(str(x))).values
    
    batch_ids = []
    psi_values = []
    null_rates = []
    mean_lengths = []
    
    for batch in stream_data:
        b_id = batch['batch_id'].iloc[0]
        lengths = batch['string'].apply(lambda x: len(str(x))).values
        
        # PSI
        try:
            psi = calculate_psi(ref_lengths, lengths)
        except:
            psi = 0.0
            
        # Null rate in sectionName
        null_rate = batch['sectionName'].isnull().mean()
        
        batch_ids.append(b_id)
        psi_values.append(psi)
        null_rates.append(null_rate)
        mean_lengths.append(np.mean(lengths))
        
        # Alerting Logic
        if psi > 0.2:
            print(f"ALERT: Significant Drift Detected in Batch {b_id}! PSI={psi:.3f}")
        if null_rate > 0.1:
            print(f"ALERT: High Null Rate Detected in Batch {b_id}! Null Rate={null_rate*100:.1f}%")

    # Render Dashboard
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Production Monitoring Dashboard', fontsize=16)
    
    # 1. PSI over time
    axs[0].plot(batch_ids, psi_values, marker='o', color='red')
    axs[0].axhline(y=0.2, color='black', linestyle='--', label='Drift Alert Threshold')
    axs[0].set_title('Feature Drift (Text Length PSI)')
    axs[0].set_ylabel('PSI')
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Null Rate over time
    axs[1].bar(batch_ids, null_rates, color='orange')
    axs[1].axhline(y=0.1, color='black', linestyle='--', label='Null Alert Threshold')
    axs[1].set_title('Data Quality: SectionName Null Rate')
    axs[1].set_ylabel('Null Fraction')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. Mean Length over time
    axs[2].plot(batch_ids, mean_lengths, marker='s', color='blue')
    axs[2].set_title('Distribution Stats: Mean Text Length')
    axs[2].set_xlabel('Batch ID (Time)')
    axs[2].set_ylabel('Mean Char Length')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path)
    plt.close()
    print(f"Dashboard saved to {output_path}")

def main():
    print("Initializing Monitoring Pipeline...")
    os.makedirs('results', exist_ok=True)
    
    # Load resplit val data to act as our simulated production stream
    dev_path = "data/raw/scicite/scicite/resplit_val.jsonl"
    if not os.path.exists(dev_path):
        dev_path = "data/raw/scicite/scicite/dev.jsonl"
    df_prod = pd.read_json(dev_path, lines=True)

    # Use resplit test data as the "reference" distribution
    test_path = "data/raw/scicite/scicite/resplit_test.jsonl"
    if not os.path.exists(test_path):
        test_path = "data/raw/scicite/scicite/test.jsonl"
    df_ref = pd.read_json(test_path, lines=True)
    
    print("Simulating stream over 5 batches...")
    stream = simulate_stream(df_prod, batches=5)
    
    # Save simulated mock logs
    mock_log_path = "results/monitoring_held_out_logs.json"
    pd.concat(stream).to_json(mock_log_path, orient='records', lines=True)
    print(f"Saved simulated production logs to {mock_log_path}")
    
    print("Running Drift Detection & Generating Dashboard...")
    generate_dashboard(stream, df_ref, "results/monitoring_dashboard.png")

if __name__ == "__main__":
    main()
