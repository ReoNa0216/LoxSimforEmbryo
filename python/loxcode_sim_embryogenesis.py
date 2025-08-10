import numpy as np
import random
import collections
from collections import deque
import time as py_time  
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm  


class LoxCodeEmbryoModel:
    """
    基于随机主体的胚胎发育模型，用于模拟LoxCode条形码和细胞命运确定过程
    """
    def __init__(self):
        # 默认参数
        self.times = np.array([132.241, 25.2327, 2.77537, 3.43722, 8.35813, 18.8558, 
                              0.787869, 3.86622, 3.94177, 1.66911, 8.32118, 18.9715, 
                              19.9047, 0.0391616, 32.2828, 9.06088])
        self.fluxes = np.array([21.8959, 71.6627, 30.7494, 55.6853, 53.9915, 31.0197, 
                               37.6783, 55.2224, 45.9964, 57.3218, 43.7055, 46.1781, 
                               72.587, 40.2534, 47.2364, 36.0463]) / 100.0  # 转换为0-1概率
        self.diff_speed = 1.0 / 6.63608
        self.lox_original = list(range(1, 14))  # 原始条形码序列[1,2,3,...,13]
        
        # 计算分裂时间
        self.split_times = self._calculate_split_times()
        
        # 创建转换矩阵
        self.transition_matrix = self._create_transition_matrix()
        
    def _calculate_split_times(self):
        """计算每种细胞类型的分裂时间"""
        split_times = np.zeros(16)
        split_times[0] = self.times[0]
        split_times[1] = split_times[0] + self.times[1]
        split_times[2] = split_times[1] + self.times[2]
        split_times[3] = split_times[2] + self.times[3]
        split_times[4] = split_times[2] + self.times[4]
        split_times[5] = split_times[1] + self.times[5]
        split_times[6] = split_times[5] + self.times[6]
        split_times[7] = split_times[6] + self.times[7]
        split_times[8] = split_times[6] + self.times[8]
        split_times[9] = split_times[5] + self.times[9]
        split_times[10] = split_times[9] + self.times[10]
        split_times[11] = split_times[10] + self.times[11]
        split_times[12] = split_times[10] + self.times[12]
        split_times[13] = split_times[9] + self.times[13]
        split_times[14] = split_times[13] + self.times[14]
        split_times[15] = split_times[13] + self.times[15]
        return split_times
    
    def _create_transition_matrix(self):
        """创建细胞命运转换矩阵"""
        dim = 33
        D = np.zeros((dim, dim))
        
        # 根据fluxes参数设置转换概率
        D[0, 1] = self.fluxes[0]
        D[0, 16] = 1 - D[0, 1]
        D[1, 2] = self.fluxes[1]
        D[1, 5] = 1 - D[1, 2]
        D[2, 3] = self.fluxes[2]
        D[2, 4] = 1 - D[2, 3]
        D[3, 17] = self.fluxes[3]
        D[3, 18] = 1 - D[3, 17]
        D[4, 19] = self.fluxes[4]
        D[4, 20] = 1 - D[4, 19]
        D[5, 6] = self.fluxes[5]
        D[5, 9] = 1 - D[5, 6]
        D[6, 7] = self.fluxes[6]
        D[6, 8] = 1 - D[6, 7]
        D[7, 21] = self.fluxes[7]
        D[7, 22] = 1 - D[7, 21]
        D[8, 23] = self.fluxes[8]
        D[8, 24] = 1 - D[8, 23]
        D[9, 10] = self.fluxes[9]
        D[9, 13] = 1 - D[9, 10]
        D[10, 11] = self.fluxes[10]
        D[10, 12] = 1 - D[10, 11]
        D[11, 25] = self.fluxes[11]
        D[11, 26] = 1 - D[11, 25]
        D[12, 27] = self.fluxes[12]
        D[12, 28] = 1 - D[12, 27]
        D[13, 14] = self.fluxes[13]
        D[13, 15] = 1 - D[13, 14]
        D[14, 29] = self.fluxes[14]
        D[14, 30] = 1 - D[14, 29]
        D[15, 31] = self.fluxes[15]
        D[15, 32] = 1 - D[15, 31]
        
        return D

    def is_odd(self, a):
        """判断一个数是否为奇数"""
        return a % 2 != 0
    
    def generate_codes(self, N, P, lox_size):
        """
        生成LoxCode条形码
        
        参数:
        N: 需要生成的条形码数量
        P: 重组事件的平均次数
        lox_size: loxP位点数量
        
        返回:
        生成的条形码列表
        """
        min_length = 82
        lox = 34
        elem = 7
        
        np.random.seed(int(py_time.time()))
        codes = {}
        multi_codes = []
        
        for _ in range(N[0]):
            cassette = list(range(1, lox_size + 1))
            n_rec = P[0]
            actual_rec = 0
            stored = False
            
            for _ in range(int(n_rec) + 1):
                if len(cassette) == 1:
                    code_tuple = tuple(cassette)
                    if code_tuple in codes:
                        codes[code_tuple] += 1
                    else:
                        codes[code_tuple] = 1
                    multi_codes.append(cassette)
                    stored = True
                    break
                
                if (len(cassette) - 2) * lox + (len(cassette) - 1) * elem < min_length:
                    code_tuple = tuple(cassette)
                    if code_tuple in codes:
                        codes[code_tuple] += 1
                    else:
                        codes[code_tuple] = 1
                    multi_codes.append(cassette)
                    stored = True
                    break
                
                pos1 = np.random.randint(0, len(cassette))
                pos2 = np.random.randint(0, len(cassette))
                
                if abs(pos1 - pos2) * lox + (abs(pos1 - pos2) + 1) * elem < min_length:
                    continue
                
                if self.is_odd(pos1) == self.is_odd(pos2):
                    dice = np.random.random()
                    if dice < 1:
                        # 反转区间内的元素并将它们变为负数
                        for j in range(min(pos1, pos2), max(pos1, pos2) + 1):
                            cassette[j] *= -1
                        # 反转序列
                        cassette[min(pos1, pos2):max(pos1, pos2) + 1] = cassette[min(pos1, pos2):max(pos1, pos2) + 1][::-1]
                        actual_rec += 1
                    else:
                        continue
                else:
                    # 切除区间内的元素
                    cassette = cassette[:min(pos1, pos2)] + cassette[max(pos1, pos2) + 1:]
                    actual_rec += 1
            
            if not stored:
                code_tuple = tuple(cassette)
                if code_tuple in codes:
                    codes[code_tuple] += 1
                else:
                    codes[code_tuple] = 1
                multi_codes.append(cassette)
                
        return multi_codes

    def sim_single(self, t_barcoding=132, t_collection=300,
                pedigree_file="loxcode_full_pedigree.csv",
                barcoding_file="loxcode_census_at_barcoding.csv",
                sampling_file="loxcode_census_at_sampling.csv"):
        """
        模拟单个胚胎的发育过程，从4细胞阶段到指定时间
        
        参数:
        t_barcoding: 条形码标记时间（小时）
        t_collection: 采样时间（小时）
        pedigree_file: 谱系文件名
        barcoding_file: 条形码化时刻普查文件名
        sampling_file: 采样时刻普查文件名
        """
        # 定义生成计数器参数
        h = 1.0  # 存活率
        
        # 定义分裂时间参数
        time_factor = 0.95
        m = 12 * time_factor
        s = 2.54 * time_factor
        sigma = np.log(1 + (s/m)**2)
        mu = np.log(m) - 0.5 * sigma
        
        # 初始化随机数生成器
        np.random.seed(int(py_time.time()))
        
        # 计算概率分布
        diff_probs = [[] for _ in range(self.transition_matrix.shape[0])]
        for i in range(self.transition_matrix.shape[0]):
            for j in range(self.transition_matrix.shape[1]):
                diff_probs[i].append(self.transition_matrix[i, j])
        
        # 初始化种群
        # 使用字典存储细胞，键为时间，值为条形码和命运路径的对
        population = {}
        
        # 设置初始时间和下一个分裂时间
        time = 40
        next_split = 40 + 0.5
        
        # 添加初始细胞
        div_time = lambda: time + np.random.lognormal(mu, np.sqrt(sigma))
        
        c1 = deque([1, 1, -1, 0])
        c2 = deque([0, 1, -1, 0])
        c3 = deque([1, 0, -1, 0])
        c4 = deque([0, 0, -1, 0])
        
        population[div_time()] = (self.lox_original.copy(), c1)
        population[div_time()] = (self.lox_original.copy(), c2)
        population[div_time()] = (self.lox_original.copy(), c3)
        population[div_time()] = (self.lox_original.copy(), c4)
        
        # 创建输出文件
        f_pedigree = open(pedigree_file, "w")
        f_barcoding = open(barcoding_file, "w")
        f_collection = open(sampling_file, "w")
        
        # 写入表头
        f_barcoding.write("loxcode, binary pedigree, fate path\n")
        f_collection.write("loxcode, binary pedigree, fate path\n")
        
        # 主模拟循环
        codes = []
        round_num = 0
        
        print(f"开始模拟，条形码时间: {t_barcoding}小时，采样时间: {t_collection}小时")
        
        # 在条形码前的时间点创建进度标记
        time_markers = [40, 60, 80, 100, t_barcoding-10, t_barcoding]
        if t_barcoding < t_collection:
            # 在条形码后添加额外的时间点
            additional_markers = [t_barcoding + (t_collection-t_barcoding)/4,
                                t_barcoding + (t_collection-t_barcoding)/2,
                                t_barcoding + 3*(t_collection-t_barcoding)/4]
            time_markers.extend(additional_markers)
        
        next_marker_idx = 0
        
        while True:
            round_num += 1
            if round_num % 1000 == 0:
                print(f"模拟第{round_num}轮，当前时间: {time:.2f}小时，种群大小: {len(population)}")
                
            if len(population) == 0:
                print("种群灭绝，模拟结束")
                break
                
            # 获取下一个分裂的细胞
            next_time = min(population.keys())
            time = next_time
            loxcode, fate_path = population[next_time]
            del population[next_time]
            
            # 时间进度报告
            if next_marker_idx < len(time_markers) and time >= time_markers[next_marker_idx]:
                progress = min(100, int((time / t_collection) * 100))
                print(f"📊 模拟进度: {progress}% - 时间: {time:.1f}小时, 种群大小: {len(population)}")
                next_marker_idx += 1
            
            # 细胞分裂或死亡
            if np.random.random() < h:  # 细胞分裂
                # 记录谱系
                f_pedigree.write(f"{time},")
                for p in fate_path:
                    f_pedigree.write(f"{p} ")
                f_pedigree.write("\n")
                
                # 创建子细胞命运路径
                ped0 = deque(fate_path)
                ped1 = deque(fate_path)
                ped0.appendleft(0)
                ped1.appendleft(1)
                
                # 决定分裂时间函数
                if fate_path[-1] != 16:  # 非血液细胞
                    div_time_func = div_time
                else:  # 血液细胞分裂更慢
                    div_time_func = lambda: time + np.random.lognormal(1.3*mu, 1.3*np.sqrt(sigma))
                
                # 添加两个子细胞
                population[div_time_func()] = (loxcode.copy(), ped0)
                population[div_time_func()] = (loxcode.copy(), ped1)
            
            # 条形码标记
            if time > t_barcoding and t_barcoding != -1:
                N = [len(population)]
                P = [4]  # 每个细胞平均4次重组
                cassette_size = 13
                codes = self.generate_codes(N, P, cassette_size)
                
                # 为每个细胞分配条形码
                counter = 0
                for cell_time in sorted(population.keys()):
                    loxcode, fate = population[cell_time]
                    population[cell_time] = (codes[counter].copy(), fate)
                    counter += 1
                
                # 条形码化时刻的人口普查
                for cell_time, (loxcode, fate_path) in population.items():
                    f_barcoding.write(" ".join(map(str, loxcode)) + ",")
                    for p in fate_path:
                        if p == -1:
                            f_barcoding.write(",")
                        else:
                            f_barcoding.write(f"{p} ")
                    f_barcoding.write("\n")
                
                # 标记完成后设置为-1，避免重复标记
                t_barcoding = -1
                print(f"条形码标记完成，当前时间: {time:.2f}小时，种群大小: {len(population)}")
            
            # 命运决定
            if time > next_split:
                for cell_time in list(population.keys()):
                    loxcode, fate_path = population[cell_time]
                    current_state = fate_path[-1]
                    
                    if current_state >= 16:  # 已经是终端细胞类型
                        continue
                    
                    if self.split_times[current_state] < time and np.random.random() < self.diff_speed:
                        # 根据转移概率选择新的状态
                        probs = diff_probs[current_state]
                        diff = np.random.choice(range(len(probs)), p=probs)
                        fate_path.append(diff)
                        population[cell_time] = (loxcode, fate_path)
                
                next_split = time + 1
            
            # 结束条件
            if time > t_collection:
                print(f"达到采样时间，模拟结束，最终种群大小: {len(population)}")
                # 记录采样时的种群状态
                for cell_time, (loxcode, fate_path) in population.items():
                    f_collection.write(" ".join(map(str, loxcode)) + ",")
                    for p in fate_path:
                        if p == -1:
                            f_collection.write(",")
                        else:
                            f_collection.write(f"{p} ")
                    f_collection.write("\n")
                break
        
        # 关闭文件
        f_pedigree.close()
        f_barcoding.close()
        f_collection.close()
        
        print("模拟完成，结果已写入文件")

    def sim_multiple(self, num_embryos=3, t_barcoding=132, t_collection=300):
        """
        模拟多个胚胎的发育过程
        
        参数:
        num_embryos: 要模拟的胚胎数量
        t_barcoding: 条形码标记时间（小时）
        t_collection: 采样时间（小时）
        """
        embryo_files = []
        
        # 添加进度条
        for i in tqdm(range(num_embryos), desc="模拟胚胎进度", unit="embryo"):
            print(f"\n开始模拟胚胎 #{i+1}/{num_embryos}")
            # 为每个胚胎创建唯一的输出文件名
            output_suffix = f"_embryo{i+1}"
            pedigree_file = f"loxcode_full_pedigree{output_suffix}.csv"
            barcoding_file = f"loxcode_census_at_barcoding{output_suffix}.csv"
            sampling_file = f"loxcode_census_at_sampling{output_suffix}.csv"
            
            # 模拟单个胚胎，但使用不同的输出文件
            self.sim_single(t_barcoding, t_collection, 
                            pedigree_file=pedigree_file,
                            barcoding_file=barcoding_file,
                            sampling_file=sampling_file)
            
            # 记录采样文件名以供后续分析
            embryo_files.append(sampling_file)
        
        return embryo_files
    
    def analyze_results(self, sampling_files=None, output_prefix="results"):
        """
        分析采样结果，生成组织-条形码矩阵并计算组织间相似性
        
        参数:
        sampling_files: 采样数据文件名列表，如果为None则使用默认文件
        output_prefix: 输出文件前缀
        """
        import pandas as pd
        import numpy as np
        from scipy.spatial.distance import cosine
        
        print("开始分析采样结果...")
        
        if sampling_files is None:
            sampling_files = ["loxcode_census_at_sampling.csv"]
        
        # 处理每个胚胎的数据
        all_embryo_data = []
        for file_idx, sampling_file in enumerate(tqdm(sampling_files, desc="读取胚胎数据")):
            print(f"处理胚胎 #{file_idx+1} 数据: {sampling_file}")
            embryo_data = []
            
            # 读取采样数据
            with open(sampling_file, 'r') as f:
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                        
                    loxcode = tuple(map(int, parts[0].strip().split()))
                    
                    # 提取命运路径中的最后一个状态（终端细胞类型）
                    fate_parts = parts[-1].strip().split()
                    if len(fate_parts) > 0:
                        tissue_type = int(fate_parts[-1])
                        # 只关注16-32的终端细胞类型
                        if tissue_type >= 16:
                            embryo_data.append((loxcode, tissue_type))
            
            all_embryo_data.append(embryo_data)
        
        # 定义组织类型名称映射
        tissue_names = {
            16: "blood",
            17: "L brain I", 18: "R brain I",
            19: "L brain III", 20: "R brain III",
            21: "L gonad", 22: "R gonad",
            23: "L kidney", 24: "R kidney",
            25: "L foot", 26: "L leg",
            27: "R foot", 28: "R leg",
            29: "L hand", 30: "L arm",
            31: "R hand", 32: "R arm"
        }
        
        # 为每个胚胎创建组织-条形码计数矩阵
        embryo_matrices = []
        for embryo_idx, embryo_data in enumerate(all_embryo_data):
            # 创建组织-条形码计数矩阵
            tissue_barcode_counts = {}
            for tissue_id in range(16, 33):
                tissue_barcode_counts[tissue_id] = collections.Counter()
            
            for barcode, tissue_id in embryo_data:
                tissue_barcode_counts[tissue_id][barcode] += 1
            
            # 获取所有独特条形码
            all_barcodes = set()
            for tissue_id, counter in tissue_barcode_counts.items():
                all_barcodes.update(counter.keys())
            all_barcodes = list(all_barcodes)
            
            # 创建组织-条形码矩阵
            matrix_data = []
            tissue_ids = sorted(tissue_barcode_counts.keys())
            
            for tissue_id in tissue_ids:
                row = [tissue_barcode_counts[tissue_id][barcode] for barcode in all_barcodes]
                matrix_data.append(row)
            
            # 创建DataFrame
            barcode_labels = [str(b) for b in all_barcodes]
            tissue_labels = [tissue_names[tid] for tid in tissue_ids]
            barcode_matrix = pd.DataFrame(matrix_data, index=tissue_labels, columns=barcode_labels)
            
            # 保存该胚胎的原始组织-条形码矩阵
            barcode_matrix.to_csv(f"{output_prefix}_embryo{embryo_idx+1}_tissue_barcode_matrix.csv")
            
            # log(1+x)转换
            log_barcode_matrix = np.log1p(barcode_matrix)
            
            embryo_matrices.append(log_barcode_matrix)
        
        # 计算每个胚胎的组织间余弦相似度矩阵
        similarity_matrices = []
        
        for embryo_idx, log_barcode_matrix in enumerate(tqdm(embryo_matrices, desc="计算相似度矩阵")):
            num_tissues = len(log_barcode_matrix)
            similarity_matrix = np.zeros((num_tissues, num_tissues))
            
            # 使用log转换后的数据计算余弦相似度
            for i in range(num_tissues):
                for j in range(num_tissues):
                    vec1 = np.array(log_barcode_matrix.iloc[i])
                    vec2 = np.array(log_barcode_matrix.iloc[j])
                    
                    # 避免除零错误
                    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
                        similarity_matrix[i, j] = 0
                    else:
                        # 计算余弦相似度 (1 - cosine距离)
                        similarity_matrix[i, j] = 1 - cosine(vec1, vec2)
            
            similarity_matrices.append(similarity_matrix)
        
        # 计算平均相似度矩阵（MCSA矩阵）
        avg_similarity_matrix = np.mean(similarity_matrices, axis=0)
        
        # 归一化相似度矩阵到0-1范围
        if np.max(avg_similarity_matrix) > np.min(avg_similarity_matrix):  # 避免除以零
            avg_similarity_matrix = (avg_similarity_matrix - np.min(avg_similarity_matrix)) / (np.max(avg_similarity_matrix) - np.min(avg_similarity_matrix))
        
        # 创建相似度矩阵DataFrame
        tissue_labels = [tissue_names[tid] for tid in sorted(tissue_names.keys())]
        avg_similarity_df = pd.DataFrame(avg_similarity_matrix, index=tissue_labels, columns=tissue_labels)
        avg_similarity_df.to_csv(f"{output_prefix}_mcsa_matrix.csv")
        print(f"平均组织相似度矩阵(MCSA)已保存至 {output_prefix}_mcsa_matrix.csv")
        
        # 返回结果数据
        return {
            'embryo_matrices': embryo_matrices,
            'similarity_matrices': similarity_matrices,
            'mcsa_matrix': avg_similarity_df,
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='模拟胚胎发育和LoxCode条形码过程')
    parser.add_argument('--barcoding', type=float, default=132, help='条形码标记时间(小时)')
    parser.add_argument('--collection', type=float, default=300, help='采样时间(小时)')
    parser.add_argument('--num_embryos', '-n', type=int, default=3, help='要模拟的胚胎数量')
    parser.add_argument('--analyze', action='store_true', help='分析结果并生成相似性矩阵')
    args = parser.parse_args()
    
    model = LoxCodeEmbryoModel()
    
    # 模拟多个胚胎
    if args.num_embryos > 1:
        sampling_files = model.sim_multiple(args.num_embryos, args.barcoding, args.collection)
        if args.analyze:
            model.analyze_results(sampling_files)
    else:
        # 保持原有的单胚胎模拟行为
        model.sim_single(args.barcoding, args.collection)
        if args.analyze:
            model.analyze_results()

if __name__ == "__main__":
    main()

