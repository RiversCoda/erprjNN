# dataprocess\addNoise.py

import json
import os
import glob
import scipy.io
import numpy as np
from datetime import datetime
import shutil
import random

def extract_name(file_path):
    # Normalize path separators
    file_path = os.path.normpath(file_path)
    parts = file_path.split(os.sep)
    if 'p4-test' in parts:
        idx = parts.index('p4-test')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ''

# Load configuration
config_file = 'dataprocess\\addNoiseText.JSON'
with open(config_file, 'r') as f:
    config = json.load(f)

quiet_data_paths = config['quiet_data_paths']
noise_data_paths = config['noise_data_paths']
window_size = config['window_size']
step_size = config['step_size']
noise_multiplier = config['noise_multiplier']

# Output directories
today = datetime.now().strftime('%Y%m%d')
output_dir = os.path.join('addNoise_data', today)
noise_output_dir = os.path.join(output_dir, 'noise')
raw_output_dir = os.path.join(output_dir, 'raw')

# Make sure output directories exist
os.makedirs(noise_output_dir, exist_ok=True)
os.makedirs(raw_output_dir, exist_ok=True)

# Copy the configuration file to the output directory with timestamp
config_backup_file = os.path.join(output_dir, 'addNoiseTextCopy.JSON')
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with open(config_file, 'r') as f:
    config_content = json.load(f)
config_content['timestamp'] = timestamp
with open(config_backup_file, 'w') as f:
    json.dump(config_content, f, indent=4, ensure_ascii=False)

# Collect all quiet data files
quiet_files = []
for path in quiet_data_paths:
    files = glob.glob(path, recursive=True)
    quiet_files.extend(files)

# Collect all noise data files
noise_files = []
for path in noise_data_paths:
    files = glob.glob(path, recursive=True)
    noise_files.extend(files)

# Ensure we have noise files
if not noise_files:
    raise ValueError("No noise data files found.")

# Process quiet data files
for quiet_file in quiet_files:
    # Load quiet data
    data = scipy.io.loadmat(quiet_file)
    if 'accresult' not in data:
        continue  # Skip if 'accresult' variable is not in the file
    accresult = data['accresult']  # shape (4, 5000)
    total_length = accresult.shape[1]
    # Sliding window
    num_slices = (total_length - window_size) // step_size + 1
    if num_slices <= 0:
        continue  # Skip if data is too short
    # Extract 'name' from quiet_file path
    name = extract_name(quiet_file)
    if not name:
        continue  # Skip if 'name' cannot be determined
    for i in range(num_slices):
        start = i * step_size
        end = start + window_size
        slice_data = accresult[:, start:end]  # shape (4, window_size)
        # Save the quiet data slice once
        # Determine output file name
        base_name = os.path.basename(quiet_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        slice_name = f"{base_name_no_ext}_slice{i}.mat"
        raw_output_subdir = os.path.join(raw_output_dir, name, 'scg')
        raw_output_file = os.path.join(raw_output_subdir, slice_name)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(raw_output_file), exist_ok=True)
        # Save the quiet data slice
        scipy.io.savemat(raw_output_file, {'accresult': slice_data})

        # For noise_multiplier times, generate noisy data
        for n in range(noise_multiplier):
            # Randomly select a noise file
            noise_file = random.choice(noise_files)
            # Load noise data
            noise_data = scipy.io.loadmat(noise_file)
            if 'accresult' not in noise_data:
                continue  # Skip if 'accresult' variable is not in the file
            noise_accresult = noise_data['accresult']  # shape (4, 5000)
            noise_total_length = noise_accresult.shape[1]
            if noise_total_length < window_size:
                continue  # Skip this noise file if it's too short
            # Randomly select a starting point
            noise_start_idx = random.randint(0, noise_total_length - window_size)
            noise_slice = noise_accresult[:, noise_start_idx:noise_start_idx+window_size]  # shape (4, window_size)
            # Add the noise slice to the quiet data slice
            noisy_slice = slice_data + noise_slice
            # Save the noisy data
            noisy_slice_name = f"{base_name_no_ext}_slice{i}_noise{n}.mat"
            noisy_output_subdir = os.path.join(noise_output_dir, name, 'scg')
            noisy_output_file = os.path.join(noisy_output_subdir, noisy_slice_name)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(noisy_output_file), exist_ok=True)
            # Save the noisy data with the path of the noise data
            scipy.io.savemat(noisy_output_file, {
                'accresult': noisy_slice,
                'noise_source': noise_file
            })
