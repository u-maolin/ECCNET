# ECCNET: Deep Learning for eccDNA Prediction

This project uses deep learning techniques to predict extrachromosomal circular DNA (eccDNA) from sequencing data.

## Project Structure

- **data**: Contains input data and processed data.
- **models**: Contains trained models and model definition files.
- **utils**: Contains utility scripts for data processing and visualization.
  - **check_shape.py**: Contains functions to check the shape of .npy files.
  - **data_generator.py**: Contains the data generator class for training the model.
  - **file_preprocess.py**: Contains functions to process large file lists.
  - **preprocess.py**: Contains functions for preprocessing input data, including encoding sequences and generating negative samples.
  - **visualization.py**: Contains functions for visualizing training results, including plotting loss, accuracy, confusion matrix, and ROC curve.
  - **prediction.py**: Contains the function for making predictions on encoded input data.
- **train.py**: Main script for training the model.
- **predict.py**: Main script for predicting new samples.
- **requirements.txt**: Lists the required Python packages.
- **README.md**: Provides project description, installation, and usage instructions.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/eccnet.git
    cd eccnet
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run:
```bash
python train.py -ecc data/train/inputs/eccdna.bed -g data/genome/human/hg19/hg19.fa -o data/train/outputs -m my_model -e 10 -c data/genome/human/hg19/hg19.chrom.sizes -t extract --seed 42
Prediction
To predict new samples, run:


python predict.py -ecc data/prediction/inputs/prediction.bed -g data/genome/human/hg19/hg19.fa -o data/prediction/outputs -m models/model/my_model/my_model.h5 -t 20bp
Visualization
To visualize the training results, run:

# Example command to generate plots
python utils/visualization.py -i data/train/outputs -o data/visualization/outputs

Parameters Description
-ecc: Path to the eccDNA BED file.
-g: Path to the genome FASTA file.
-o: Output directory for the results.
-m: Path to save or load the model file.
-e: Number of training epochs.
-c: Path to the genome Chrom_size file.
-t: Encoding type used during training or prediction (extract or circle).
--seed: Seed for random number generation.

License
This project is licensed under the MIT License. See the LICENSE file for details.
