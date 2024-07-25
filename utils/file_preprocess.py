# utils/file_processing.py

def process_large_file_list(file_list, batch_size):
    for i in range(0, len(file_list), batch_size):
        batch_file_paths = file_list[i:i + batch_size]
        yield batch_file_paths

