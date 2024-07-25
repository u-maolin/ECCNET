# utils/check_shape.py

import os
import numpy as np

def check_npy_shape(data_folder, expected_shape=(20, 4), delete_inconsistent=False):
    consistent = True  # 是否一致的标志

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                if data.shape != expected_shape:
                    print(f"Shape mismatch in file: {file_path}. Expected: {expected_shape}, Actual: {data.shape}")
                    consistent = False

                    if delete_inconsistent:
                        os.remove(file_path)
                        print(f"File removed: {file_path}")

    if consistent:
        print("All .npy files have the expected shape.")
    else:
        print("Some .npy files have inconsistent shapes.")

def new_check_npy_shape(data_folder, expected_shape=(1000, 4), delete_inconsistent=False):
    consistent = True  # 是否一致的标志

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                if data.shape != expected_shape:
                    print(f"Shape mismatch in file: {file_path}. Expected: {expected_shape}, Actual: {data.shape}")
                    consistent = False

                    if delete_inconsistent:
                        os.remove(file_path)
                        print(f"File removed: {file_path}")

    if consistent:
        print("All .npy files have the expected shape.")
    else:
        print("Some .npy files have inconsistent shapes.")

