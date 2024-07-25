import os
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
import random
from .check_shape import check_npy_shape,new_check_npy_shape
import re
import math
import random
import time

def extract_sequence(chromosome, start, end, genome_fasta_path):
    fasta = Fasta(genome_fasta_path)
    sequence = str(fasta[chromosome][start-1:end])
    fasta.close()
    return sequence


def one_hot_encode_sequence(sequence):
    sequence = sequence.upper()  # 将序列转换为大写
    bases = 'ACGT'
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)  # 使用 float32 数据类型

    for i, base in enumerate(sequence):
        if base in bases:
            encoding[i, bases.index(base)] = 1
        else:
            encoding[i, :] = [0.25, 0.25, 0.25, 0.25]  # 平均分配概率

    return encoding



def preprocess_bed_file(input_bed_path, output_bed_path):
    with open(input_bed_path, 'r') as input_file, open(output_bed_path, 'w') as output_file:
        for line in input_file:
            parts = line.strip().split('\t')
            # Check if the line has at least three columns
            if len(parts) >= 3:
                # 从染色体名字中删除非字母数字字符
                parts[0] = re.sub(r'\W', '', parts[0])
                output_file.write('\t'.join(parts) + '\n')
            else:
                print("Warning: Skipping line due to insufficient columns:", line)


def encode_bed_file(bed_file_path, genome_fasta_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load genome FASTA file
    genome = Fasta(genome_fasta_path)

    # Read bed file and encode sequences
    with open(bed_file_path, 'r') as bed_file:
        #修改，如果第一行为注释行则跳过
        next(bed_file)
        for line in tqdm(bed_file, desc="正在编码序列"):
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            # Extract upstream and downstream sequences from genome
            if start-5 < 0:
                upstream_sequence = str(genome[chrom][1:11])
            else:
                upstream_sequence = str(genome[chrom][start-5:start+5])
            if end+5>len(genome[chrom]):
                downstream_sequence = str(genome[chrom][len(genome[chrom])-10:len(genome[chrom])])
            else:
                downstream_sequence = str(genome[chrom][end-5:end+5])
            
            # Make sure the sequences are of length 10
            upstream_sequence = upstream_sequence[:10]
            downstream_sequence = downstream_sequence[:10]

            # Combine upstream and downstream sequences
            sequence = upstream_sequence + downstream_sequence

            # One-hot encode sequence
            encoding = one_hot_encode_sequence(sequence)

            # Save encoding as npy file
            sample_name = f"{chrom}_{start}_{end}"
            npy_file_path = os.path.join(output_folder, f"{sample_name}.npy")
            np.save(npy_file_path, encoding)

def save_as_npy(array, output_folder, sample_index):
    filename = f"{output_folder}/sample_{sample_index}.npy"
    np.save(filename, array)


def circle_encode_tocenter(bed_file_path, genome_fasta_path, output_folder, max_length=1000):
    data_list = []

    with open(bed_file_path, 'r') as bed_file:
        for sample_index,line in enumerate(bed_file, start=1):
            # 确保bed格式
            chromosome, start, end = line.strip().split('\t')
            start, end = int(start), int(end)

            # 提取序列和独热编码
            sequence = extract_sequence(chromosome, start, end, genome_fasta_path)
            encoding = one_hot_encode_sequence(sequence)
            eccDNA_length = end - start + 1
            multiples = max_length // eccDNA_length

            # 重复编码multiples次
            repeated_encoding = np.tile(encoding, (multiples,1))


            # 计算循环后的长度
            repeated_length = len(repeated_encoding)
            # 如果循环后的长度超过了 max_length，截取前 max_length 行
            if repeated_length > max_length:
                repeated_encoding = repeated_encoding[:max_length]


            # 计算剩余需要填充的区域
            remaining_space = max_length - multiples * eccDNA_length
            remaining_space_half = math.floor(remaining_space / 2)

            rows_on_left_side = remaining_space_half + remaining_space % 2
            rows_on_right_side = remaining_space_half

            # 合并编码结果
            result_encoding = np.zeros((max_length, 4))  # 固定编码shape，用0填充

            # 根据需要填充
            result_encoding[:rows_on_left_side] = encoding[:rows_on_left_side]
            result_encoding[-rows_on_right_side:] = encoding[-rows_on_right_side:]
            result_encoding[rows_on_left_side:-rows_on_right_side] = repeated_encoding

            # 保存为npy文件
            save_as_npy(result_encodind, output_folder, sample_index)
            data_list.append(result_encoding)
        
        result_matrix = np.vstack(data_list)
        return result_matrix



def generate_negative_samples(eccdna_bed_path, chrom_sizes_path, num_negative_samples, output_bed_path, seed=None):
    # 设置随机数种子
    if seed is not None:
        random.seed(seed)

    # 加载bed文件
    eccdna_regions = []
    with open(eccdna_bed_path, 'r') as eccdna_bed:
        for line in eccdna_bed:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            eccdna_regions.append((chrom, start, end))

    # 加载染色体长度文件
    chrom_sizes = {}
    with open(chrom_sizes_path, 'r') as chrom_sizes_file:
        for line in chrom_sizes_file:
            parts = line.strip().split('\t')
            chrom = parts[0]
            size = int(parts[1])
            chrom_sizes[chrom] = size

    # 生成负样本

    negative_samples = []
    valid_chromosomes = list(chrom_sizes.keys())

    # 开始计时
    start_time = time.time()
    print("Start generating negative samples...")
    
    # 遍历要生成的负样本数量
    for i in range(num_negative_samples):
        # 每生成100个负样本就展示一次时间
        if (i + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Generated {i + 1} negative samples in {elapsed_time:.2f} seconds.")
        
        # 随机挑选染色体
        chrom1 = random.choice(valid_chromosomes)
    
        # 获取当前染色体的长度
        chrom_length = chrom_sizes[chrom1]
    
        # 重复次数，用于避免无限循环
        retries = 0
    
        # 循环直到生成不与任何 eccDNA 区域重叠的区域
        while retries < 100:  # 设置一个最大重试次数，避免无限循环
            # 随机挑选起始位点
            start = random.randint(1, chrom_length - 20)
        
            # 设置终止位点
            end = start + 20
        
            # 检查生成的区域是否与任何 eccDNA 区域重叠
            overlaps_eccdna = any(
                (start >= ecc_start and start <= ecc_end) or (end >= ecc_start and end <= ecc_end)
                for _, ecc_start, ecc_end in eccdna_regions
            )
        
            if not overlaps_eccdna:
                # 如果不重叠，则添加到负样本列表中，并跳出循环
                negative_samples.append((chrom1, start, end))
                break
        
            retries += 1  # 增加重试次数

    # 将负样本保存到输出 BED 文件中
    with open(output_bed_path, 'w') as output_bed:
        for chrom, start, end in negative_samples:
            output_bed.write(f"{chrom1}\t{start}\t{end}\n")

def new_generate_negative_samples(eccdna_bed_path, chrom_size_path, num_negative_samples, output_bed_path, seed=None):
    # 设置随机数种子
    if seed is not None:
        random.seed(seed)

    # 加载eccDNA的bed文件
    eccdna_regions = []
    with open(eccdna_bed_path, 'r') as eccdna_bed:
        for line in eccdna_bed:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            eccdna_regions.append((chrom, start, end))

    # 加载染色体大小文件
    chrom_sizes = {}
    with open(chrom_size_path, 'r') as chrom_size_file:
        for line in chrom_size_file:
            parts = line.strip().split('\t')
            chrom = parts[0]
            size = int(parts[1])
            chrom_sizes[chrom] = size

    # 生成负样本
    negative_samples = []
    valid_chromosomes = list(chrom_sizes.keys())
    print("Start generating negative samples...")
    start_time = time.time()
    for i in range(num_negative_samples):
        if (i + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Generated {i + 1} negative samples in {elapsed_time:.2f} seconds.")

        
        # 随机选择一个eccDNA正样本
        eccdna_region = random.choice(eccdna_regions)
        eccdna_chrom, eccdna_start, eccdna_end = eccdna_region
        eccdna_length = eccdna_end - eccdna_start

        # 随机选择一个染色体生成负样本
        chrom = eccdna_chrom

        # 确保负样本的长度与eccDNA正样本相似
        length_diff = random.randint(-100,100)  # 调整这个范围根据需要
        start = max(1, eccdna_start - length_diff)
        end = start + eccdna_length

        # 检查生成的区域是否与任何eccDNA正样本重叠
        overlaps_eccdna = any(
            (start >= ecc_start and start <= ecc_end) or (end >= ecc_start and end <= ecc_end)
            for _, ecc_start, ecc_end in eccdna_regions
        )

        if not overlaps_eccdna:
            negative_samples.append((chrom, start, end))
            break
    # 保存负样本到输出文件
    with open(output_bed_path, 'w') as output_bed:
        for chrom, start, end in negative_samples:
            output_bed.write(f"{chrom}\t{start}\t{end}\n")
    print("Negative samples generation completed.")

def preprocess_data(eccdna_bed_path, genome_fasta_path, positive_output_folder, negative_output_folder, chrom_size_path, num_negative_samples, seed):
    
    # 编码正样本
    encode_bed_file(eccdna_bed_path, genome_fasta_path, positive_output_folder)
    check_npy_shape(positive_output_folder, delete_inconsistent=True)


    # 生成负样本
    negative_bed_path = os.path.join(os.path.dirname(eccdna_bed_path), 'negative_samples.bed')
    generate_negative_samples(eccdna_bed_path, chrom_size_path, num_negative_samples, negative_bed_path , seed)


    # 编码负样本
    encode_bed_file(negative_bed_path, genome_fasta_path, negative_output_folder)
    check_npy_shape(negative_output_folder, delete_inconsistent=True)

def circle_preprocess_data(eccdna_bed_path, genome_fasta_path, positive_output_folder, negative_output_folder, chrom_size_path, num_negative_samples, seed, max_length):

    # 编码正样本
    circle_encode_tocenter(eccdna_bed_path, genome_fasta_path, positive_output_folder, max_length)
    new_check_npy_shape(positive_output_folder, delete_inconsistent=True)


    # 生成负样本
    negative_bed_path = os.path.join(os.path.dirname(eccdna_bed_path), 'negative_samples.bed')
    new_generate_negative_samples(eccdna_bed_path, chrom_size_path, num_negative_samples, negative_bed_path, seed)


    # 编码负样本
    circle_encode_tocenter(negative_bed_path, genome_fasta_path, negative_output_folder, max_length)
    new_check_npy_shape(negative_output_folder, delete_inconsistent=True)
    
