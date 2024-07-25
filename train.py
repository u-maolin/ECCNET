import os
import random
from sklearn.model_selection import train_test_split
from utils.data_generator import MyDataGenerator
from models.new_my_model import create_complex_model,custom_loss
from models.my_model import create_model
from utils.visualization import plot_loss_accuracy, plot_confusion_matrix
from utils.preprocess import preprocess_data,circle_preprocess_data,plot_roc_curve
from utils.file_preprocess import process_large_file_list
import argparse
from utils.check_shape import check_npy_shape
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import matplotlib

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the eccDNA prediction model.")
    parser.add_argument("-ecc", "--eccdna_bed", default="./data/train/inputs/eccdna.bed", help="Path to the eccDNA BED file.")
    parser.add_argument("-g", "--genome_fasta", default="./data/genome/human/hg19/hg19.fa", help="Path to the genome FASTA file.")
    parser.add_argument("-o", "--output_dir", default="./data/train/outputs", help="Output directory for the model and plots.")
    parser.add_argument("--num_negative_samples", type=int, default=None, help="Number of negative samples to generate.")
    parser.add_argument("-m","--model_name", default="my_model", help="Name for the saved model.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-c", "--chrom_size",default="./data/genome/human/hg19/hg19.chrom.sizes", help="Path to the genome Chrom_size file")
    parser.add_argument("-t","--encode_type",default="extract",help="Type of the encoding samles")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
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
    eccdna_bed_path = args.eccdna_bed
    eccdna_bed_basename = os.path.splitext(os.path.basename(eccdna_bed_path))[0]
    genome_fasta_path = args.genome_fasta
    output_dir = args.output_dir
    fig_output_dir=os.path.join(output_dir,"fig",eccdna_bed_basename)
    positive_output_folder = os.path.join(output_dir, 'positive',eccdna_bed_basename)
    negative_output_folder = os.path.join(output_dir, 'negative',eccdna_bed_basename)
    model_output_dir=os.path.join("./models/model",args.model_name)
    
    # 设置种子
    seed = args.seed
    encode_type = args.encode_type
    # 获取 eccdna_bed 文件的行数，作为默认的 num_negative_samples
    if args.num_negative_samples is None:
        try:
            with open(args.eccdna_bed, 'r') as bed_file:
                num_lines = sum(1 for line in bed_file)
            args.num_negative_samples = num_lines
        except FileNotFoundError:
            print(f"Error: The specified BED file '{args.eccdna_bed}' does not exist.")
            exit(1)
    num_negative_samples=args.num_negative_samples
    # 确定最大编码长度
    max_length = determine_max_length(eccdna_bed_path)

    # 检查目录是否存在
    if os.path.exists(negative_output_folder):
    # 如果存在，找到所有的npy文件
        print(f"负样本目录已存在，准备删除负样本目录下编码结果以样本之间平衡性.")
        npy_files = glob.glob(os.path.join(negative_output_folder,'*.npy'))
    
    # 删除每个找到的npy文件
        for file_path in npy_files:
            os.remove(file_path)
        print(f"负样本已删除完成，准备生成数据")    
    if os.path.exists(positive_output_folder):
    # 如果存在，找到所有的npy文件
        print(f"正样本录已存在，准备删除正样本目录下编码结果以避免同名bed文件生成过量正样本.")
        npy_files = glob.glob(os.path.join(negative_output_folder,'*.npy'))
    
    # 删除每个找到的npy文件
        for file_path in npy_files:
            os.remove(file_path)
        print(f"正样本已删除完成，准备生成数据")
    for dir_path in [positive_output_folder,negative_output_folder,fig_output_dir,model_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(args.chrom_size):
        raise FileNotFoundError(f"File not found: {args.chrom_size}")

    chrom_size_path=args.chrom_size
    # 根据用户选择的编码方式预处理数据
    if encode_type == "extract":
        preprocess_data(eccdna_bed_path, genome_fasta_path, positive_output_folder, negative_output_folder, chrom_size_path, num_negative_samples, seed)
    elif encode_type == "circle":
        circle_preprocess_data(eccdna_bed_path, genome_fasta_path, positive_output_folder, negative_output_folder, chrom_size_path, num_negative_samples, seed)


    print("正样本编码数据已保存至",positive_output_folder)
    print("负样本编码数据已保存至",negative_output_folder)
    # 读取数据
    # 创建空列表，用于存储特征矩阵和标签向量
    X_pos, X_neg = [], []
    y_pos, y_neg = [], []

    pos_dir = positive_output_folder
    neg_dir = negative_output_folder
    # 定义要加载的文件扩展名
    valid_extensions = ('.npy',)
    # 逐批次处理文件列表
    for batch_file_paths in process_large_file_list(os.listdir(pos_dir), batch_size=1000):
        for file in batch_file_paths:
            file_path = os.path.join(pos_dir, file)
            if file_path.endswith(valid_extensions):
                X_pos.append(file_path)
                y_pos.append(1)    
    
    for batch_file_paths in process_large_file_list(os.listdir(neg_dir), batch_size=1000):
        for file in batch_file_paths:
            file_path = os.path.join(neg_dir, file)
            if file_path.endswith(valid_extensions):
                X_neg.append(file_path)
                y_neg.append(0)
    # 将正负样本分别划分为训练集和临时集
    X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(X_pos, y_pos, test_size=0.1, random_state=42)
    X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split(X_neg, y_neg, test_size=0.1, random_state=42)

    # 将正负样本的训练集合并
    X_train = X_train_pos + X_train_neg
    y_train = y_train_pos + y_train_neg

    # 将正负样本的验证集合并
    X_val = X_val_pos + X_val_neg
    y_val = y_val_pos + y_val_neg

    X_test = X_val
    y_test = y_val

    # 创建数据生成器
    batch_size = 32
    train_data_generator = MyDataGenerator(X_train, y_train, batch_size)
    val_data_generator = MyDataGenerator(X_val, y_val, batch_size)
    test_data_generator = MyDataGenerator(X_test, y_test, batch_size)

    # 创建并训练模型
    print("训练开始")
    # Create and compile the model on GPU
    with tf.device('/device:GPU:0'):
        if encode_type == "circle":
            model = create_complex_model()
        else:
            model = create_model()
        model.compile(optimizer='adam',loss=custom_loss,metrics==['accuracy'])
            

    history = model.fit(train_data_generator, epochs=args.epochs, validation_data=val_data_generator)

    # 保存模型 
    model.save(os.path.join(model_output_dir, f"{args.model_name}.h5"))
    print("模型已保存至",model_output_dir)


    # 评估模型
    test_loss, test_acc = model.evaluate(test_data_generator)
    print("Test loss: {:.4f}, Test accuracy: {:.4f}".format(test_loss, test_acc))


    matplotlib.use('Agg')
    y_pred = model.predict(test_data_generator)
    y_pred_binary = (y_pred > 0.5).astype("int32")
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)

    # 可视化
    plot_loss_accuracy(history, fig_output_dir)
    plot_confusion_matrix(y_val, y_pred_binary, fig_output_dir)
    plot_roc_curve(y_val, y_pred, fig_output_dir)
    print("可视化结果已保存至", fig_output_dir)

    # 模型性能可视化
    performance_values = [test_acc, test_loss, precision, recall, f1]
    save_path = os.path.join(fig_output_dir, 'model_performance.png')
    plot_model_performance(performance_values, fig_output_dir, save_path)
    print("模型性能图已保存至", save_path)
