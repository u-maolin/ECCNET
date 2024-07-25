import os
import pandas as pd
from tqdm import tqdm
from keras.models import load_model
from .preprocess import encode_bed_file, circle_encode_tocenter
from pyfaidx import Fasta
import numpy as np

def predict_bed_file(bed_file_path, genome_fasta_path, model_path, encoded_result_path, output_folder, encode_type, max_length):
    # Load genome FASTA file
    genome = Fasta(genome_fasta_path)

    # Load the trained model
    model = load_model(model_path)
    
    # Encode the BED file
    if encode_type == 'circle':
        circle_encode_tocenter(bed_file_path, genome_fasta_path, encoded_result_path, max_length)
    else:
        encode_bed_file(bed_file_path, genome_fasta_path, encoded_result_path, encode_type, max_length)

    # Make predictions for each encoded numpy file
    predictions = []
    for npy_file in tqdm(os.listdir(encoded_result_path), desc="正在进行预测"):
        npy_file_path = os.path.join(encoded_result_path, npy_file)

        # 从文件名中提取位置信息
        location = os.path.splitext(npy_file)[0]

        encoding = np.load(npy_file_path)

        # Reshape for model prediction
        encoding = np.expand_dims(encoding, axis=0)

        # Make prediction
        prediction = model.predict(encoding)[0][0]

        # Append the prediction to the list
        predictions.append({
            "location": location,
            "Prediction": prediction
        })

    # Create a DataFrame from the predictions
    predictions_df = pd.DataFrame(predictions)

    # Save predictions to Excel file
    excel_output_path = os.path.join(output_folder, "predictions.xlsx")
    predictions_df.to_excel(excel_output_path, index=False)

    # Save predictions to text file
    text_output_path = os.path.join(output_folder, "predictions.txt")
    predictions_df.to_csv(text_output_path, sep='\t', index=False)

