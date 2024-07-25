import os
from utils.prediction import predict_bed_file
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="predict the eccDNA probability based on trained model.")
    parser.add_argument("-ecc", "--eccdna_bed", default="./data/prediction/inputs/prediction.bed", help="Path to the eccDNA BED file.")
    parser.add_argument("-g", "--genome_fasta", default="./data/genome/human/hg19/hg19.fa", help="Path to the genome FASTA file.")
    parser.add_argument("-o", "--output_dir", default="./data/prediction/outputs", help="Output directory for the prediction results")
    parser.add_argument("-m", "--model_path", default = "./models/model/my_model/my_model.h5", help="Path to model file")
    parser.add_argument("-t", "--encode_type", default="circle", help="Encoding type used during training")
    return parser.parse_args()

def determine_max_length(bed_file):
    max_length = 0
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            start = int(parts[1])
            end = int(parts[2])
            length = end - start
            if length > max_length:
                max_length = length
    return max_length

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    # 设置路径
    bed_file_path = args.eccdna_bed
    eccdna_bed_basename = os.path.splitext(os.path.basename(bed_file_path))[0]
    model_path = args.model_path
    genome_path = args.genome_fasta
    output_path = args.output_dir
    encoded_result_path = os.path.join(output_path,"encoded_result",eccdna_bed_basename)
    output_folder_path = os.path.join(output_path,eccdna_bed_basename)
    for dir_path in [encoded_result_path, output_folder_path]:
        os.makedirs(dir_path, exist_ok=True)
    # 确定最大编码长度
    max_length = determine_max_length(bed_file_path)
    # 进行预测
    print("开始预测")
    predict_bed_file(bed_file_path, genome_path, model_path, encoded_result_path, output_folder_path, args.encode_type, max_length)
    print("预测完成，结果保存在", output_folder_path)
