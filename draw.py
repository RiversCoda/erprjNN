import os
import scipy.io as sio
import matplotlib.pyplot as plt

# Define the directory where the .mat files are located
mat_files_directory = r'D:\code\epProj\collect_data\device3\p4-test\lr\scg'

# List to hold the data from the second row (index 1)
second_rows = []

# Get all .mat files in the directory
mat_files = [f for f in os.listdir(mat_files_directory) if f.endswith('.mat')]

# Process each .mat file
for file in mat_files:
    file_path = os.path.join(mat_files_directory, file)
    mat_data = sio.loadmat(file_path)
    
    # Extract accresult attribute if it exists
    if 'accresult' in mat_data:
        accresult = mat_data['accresult']
        # Add the second row (index 1) to the list
        second_rows.append(accresult[1, :])

# Plot each second row in a separate plot
num_files = len(second_rows)
fig, axes = plt.subplots(num_files, 1, figsize=(10, num_files * 2))

for i, second_row in enumerate(second_rows):
    axes[i].plot(second_row)
    axes[i].set_title(f'File {i+1}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()
