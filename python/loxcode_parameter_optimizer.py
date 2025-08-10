import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma 
from multiprocessing import Pool
import time
from scipy.spatial.distance import cosine
import os
from typing import List, Dict, Tuple

from loxcode_sim_embryogenesis import LoxCodeEmbryoModel  # 导入已有的模型类

class LoxCodeParameterOptimizer:
    """使用CMA-ES优化LoxCode胚胎模型参数"""
    
    def __init__(self, empirical_mcsa_file=None, empirical_barcode_counts_file=None):
        """
        初始化优化器
        
        参数:
        empirical_mcsa_file: 包含经验性组织相似度矩阵的文件
        empirical_barcode_counts_file: 包含经验性组织条形码计数的文件
        """
        self.model = LoxCodeEmbryoModel()
        
        # 加载经验数据，如果文件不存在则设为None
        self.empirical_mcsa = None
        self.empirical_barcode_counts = None
        
        if empirical_mcsa_file and os.path.exists(empirical_mcsa_file):
            self.empirical_mcsa = pd.read_csv(empirical_mcsa_file, index_col=0)
            
        if empirical_barcode_counts_file and os.path.exists(empirical_barcode_counts_file):
            self.empirical_barcode_counts = pd.read_csv(empirical_barcode_counts_file, index_col=0)
            
        # 定义参数边界
        self.param_names = ["times", "fluxes", "diff_speed"]
        
        # 默认的初始参数值 (来自模型)
        self.initial_params = np.concatenate([
            self.model.times,
            self.model.fluxes * 100.0,  # 恢复到百分比形式
            np.array([6.63608])  # diff_speed的倒数
        ])
        
        # 初始参数长度应该是33 (16+16+1)
        assert len(self.initial_params) == 33
        
        # 参数的边界约束
        self.lower_bounds = np.zeros(33)
        self.upper_bounds = np.zeros(33) + 200.0  # 大多数参数上限为200
        self.upper_bounds[-1] = 20.0  # diff_speed参数上限为20
        
    def objective_function(self, params):
        """
        CMA-ES优化的目标函数，计算模拟结果与经验数据之间的差异
        
        参数:
        params: 模型参数数组
        
        返回:
        loss: 损失值，表示模拟与经验数据之间的差异
        """
        # 从参数数组中提取参数
        times = params[:16]
        fluxes = params[16:32] / 100.0  # 转换为0-1之间的概率
        diff_speed = 1.0 / params[32]  # 转换为分化速率
        
        # 临时更新模型参数
        original_times = self.model.times.copy()
        original_fluxes = self.model.fluxes.copy()
        original_diff_speed = self.model.diff_speed
        
        self.model.times = times
        self.model.fluxes = fluxes
        self.model.diff_speed = diff_speed
        
        # 重新计算分裂时间和转换矩阵
        self.model.split_times = self.model._calculate_split_times()
        self.model.transition_matrix = self.model._create_transition_matrix()
        
        # 为每次评估创建唯一的输出目录
        run_id = int(time.time() * 1000) % 10000000
        output_dir = f"optimization_run_{run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 运行模拟
            t_barcoding = 132  # 固定的条形码时间
            t_collection = 250  # 评估在250小时而不是300小时，以提高性能
            self.model.sim_single(t_barcoding, t_collection)
            
            # 分析结果
            results = self.model.analyze_results(
                output_prefix=os.path.join(output_dir, "sim_results")
            )
            
            # 提取模拟的MCSA矩阵和条形码计数
            sim_mcsa = results['similarity_matrix']
            sim_barcode_matrix = results['barcode_matrix']
            
            # 计算损失函数
            loss = 0.0
            
            # 如果有经验MCSA数据，计算与模拟MCSA的差异
            if self.empirical_mcsa is not None:
                # 提取上三角矩阵元素（不包括对角线）
                upper_tri_indices = np.triu_indices(sim_mcsa.shape[0], k=1)
                sim_mcsa_upper = sim_mcsa.values[upper_tri_indices]
                emp_mcsa_upper = self.empirical_mcsa.values[upper_tri_indices]
                
                # 计算均方误差
                mcsa_mse = np.mean((sim_mcsa_upper - emp_mcsa_upper) ** 2)
                loss += mcsa_mse * 100  # 权重放大
            
            # 如果有经验条形码计数数据，计算组织间相对条形码数的差异
            if self.empirical_barcode_counts is not None:
                # 计算每个组织的条形码总数
                sim_counts_per_tissue = sim_barcode_matrix.sum(axis=1)
                emp_counts_per_tissue = self.empirical_barcode_counts.sum(axis=1)
                
                # 归一化计数
                sim_rel_counts = sim_counts_per_tissue / sim_counts_per_tissue.sum()
                emp_rel_counts = emp_counts_per_tissue / emp_counts_per_tissue.sum()
                
                # 计算均方误差
                barcode_mse = np.mean((sim_rel_counts - emp_rel_counts) ** 2)
                loss += barcode_mse * 10  # 权重放大
            
            # 如果没有经验数据，使用内部一致性作为优化目标
            if self.empirical_mcsa is None and self.empirical_barcode_counts is None:
                # 检查左右对称性 (对于成对器官)
                paired_organs = [(17, 18), (19, 20), (21, 22), (23, 24), 
                                (25, 26), (27, 28), (29, 30), (31, 32)]
                
                asymmetry_penalty = 0
                for left_idx, right_idx in paired_organs:
                    left_idx -= 16  # 调整索引（因为组织ID从16开始）
                    right_idx -= 16
                    
                    if left_idx < sim_mcsa.shape[0] and right_idx < sim_mcsa.shape[0]:
                        asymmetry_penalty += abs(
                            sim_mcsa.iloc[left_idx, right_idx] - 0.9
                        )
                
                loss += asymmetry_penalty
            
            # 打印当前评估信息
            print(f"Params: {params[:5]}... Loss: {loss:.6f}")
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            loss = 1000.0  # 高损失值表示失败的运行
        finally:
            # 恢复原始模型参数
            self.model.times = original_times
            self.model.fluxes = original_fluxes
            self.model.diff_speed = original_diff_speed
            self.model.split_times = self.model._calculate_split_times()
            self.model.transition_matrix = self.model._create_transition_matrix()
        
        return loss
    
    def _run_single_optimization(self, seed):
        """运行单次CMA-ES优化"""
        np.random.seed(seed)
        
        # 初始参数添加随机扰动
        initial_params = self.initial_params * np.random.uniform(0.8, 1.2, size=len(self.initial_params))
        
        # 初始化CMA-ES优化器
        sigma0 = 0.3  # 初始步长
        options = {
            'bounds': [self.lower_bounds, self.upper_bounds],
            'seed': seed,
            'tolfun': 1e-5,
            'popsize': 10,  # 种群大小
            'maxiter': 300,  # 最大迭代次数，对应约3000次评估
            'verb_disp': 1,  # 显示进度
        }
        
        # 运行优化
        optimizer = cma.CMAEvolutionStrategy(initial_params, sigma0, options)
        start_time = time.time()
        
        # 保存优化历史
        history = {
            'iterations': [],
            'best_fitness': [],
            'best_params': [],
            'elapsed_time': []
        }
        
        # 优化循环
        iteration = 0
        try:
            while not optimizer.stop():
                solutions = optimizer.ask()
                fitness_values = [self.objective_function(sol) for sol in solutions]
                optimizer.tell(solutions, fitness_values)
                
                # 记录历史
                iteration += 1
                history['iterations'].append(iteration)
                history['best_fitness'].append(optimizer.best.f)
                history['best_params'].append(optimizer.best.x.copy())
                history['elapsed_time'].append(time.time() - start_time)
                
                # 每10次迭代保存一次结果
                if iteration % 10 == 0:
                    # 保存当前最佳参数
                    best_params = optimizer.best.x
                    param_df = pd.DataFrame({
                        'Parameter': self.param_names * 33,
                        'Value': best_params
                    })
                    param_df.to_csv(f'optimization_seed{seed}_best_params.csv')
                    
                    # 保存收敛历史
                    history_df = pd.DataFrame({
                        'Iteration': history['iterations'],
                        'Best Fitness': history['best_fitness'],
                        'Elapsed Time': history['elapsed_time']
                    })
                    history_df.to_csv(f'optimization_seed{seed}_history.csv')
                
                optimizer.disp()
        
        except KeyboardInterrupt:
            print(f"Optimization for seed {seed} manually interrupted")
        
        # 获取最佳结果
        result = {
            'seed': seed,
            'best_params': optimizer.best.x,
            'best_fitness': optimizer.best.f,
            'n_evaluations': optimizer.countevals,
            'history': history
        }
        
        # 保存最终结果
        np.savetxt(f'best_params_seed{seed}.txt', optimizer.best.x)
        
        return result
    
    def run_multiple_optimizations(self, n_runs=20, n_processes=4):
        """
        运行多次CMA-ES优化，从不同随机初始点开始
        
        参数:
        n_runs: 运行次数
        n_processes: 并行进程数
        """
        seeds = list(range(1, n_runs + 1))
        
        # 使用进程池并行运行优化
        with Pool(processes=n_processes) as pool:
            results = pool.map(self._run_single_optimization, seeds)
        
        # 按适应度排序结果
        sorted_results = sorted(results, key=lambda x: x['best_fitness'])
        
        # 选择前5个最佳结果
        best_results = sorted_results[:5]
        
        # 打印和保存结果
        print("\n--- 优化结果摘要 ---")
        print(f"总运行次数: {n_runs}")
        print(f"选择的最佳运行: {len(best_results)}")
        print("\n最佳参数集:")
        
        # 收集所有参数到DataFrame
        all_params = []
        for i, res in enumerate(best_results):
            params = res['best_params']
            all_params.append(params)
            print(f"运行 {i+1} (种子 {res['seed']}): 适应度 = {res['best_fitness']:.6f}")
        
        all_params = np.array(all_params)
        
        # 保存所有最佳参数
        param_names = ['times_' + str(i) for i in range(16)] + \
                     ['flux_' + str(i) for i in range(16)] + \
                     ['diff_speed']
        
        param_df = pd.DataFrame(all_params, columns=param_names)
        param_df.to_csv('best_parameters_summary.csv')
        
        # 绘制参数分布图
        self._plot_parameter_distributions(all_params, param_names)
        
        return best_results
    
    def _plot_parameter_distributions(self, params_array, param_names):
        """绘制最佳参数的分布"""
        n_params = params_array.shape[1]
        
        # 创建足够的子图
        fig, axes = plt.subplots(7, 5, figsize=(20, 28))
        axes = axes.flatten()
        
        # 绘制每个参数的分布
        for i in range(n_params):
            ax = axes[i] if i < len(axes) else axes[-1]
            values = params_array[:, i]
            
            # 绘制箱线图
            ax.boxplot(values, vert=False)
            
            # 添加散点
            y = np.random.normal(1, 0.04, size=len(values))
            ax.scatter(values, y, alpha=0.7)
            
            ax.set_title(param_names[i])
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # 隐藏未使用的子图
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('parameter_distributions.png', dpi=300)
        plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='优化LoxCode胚胎模型参数')
    parser.add_argument('--emp_mcsa', type=str, default=None, help='经验MCSA矩阵文件')
    parser.add_argument('--emp_barcode', type=str, default=None, help='经验条形码计数文件')
    parser.add_argument('--n_runs', type=int, default=20, help='优化运行次数')
    parser.add_argument('--n_processes', type=int, default=4, help='并行进程数')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = LoxCodeParameterOptimizer(
        empirical_mcsa_file=args.emp_mcsa,
        empirical_barcode_counts_file=args.emp_barcode
    )
    
    # 运行多次优化
    results = optimizer.run_multiple_optimizations(
        n_runs=args.n_runs,
        n_processes=args.n_processes
    )
    
    # 使用最佳参数运行完整模拟
    print("\n使用最佳参数运行验证模拟...")
    best_params = results[0]['best_params']
    
    # 提取参数
    times = best_params[:16]
    fluxes = best_params[16:32] / 100.0
    diff_speed = 1.0 / best_params[32]
    
    # 更新模型参数
    model = LoxCodeEmbryoModel()
    model.times = times
    model.fluxes = fluxes
    model.diff_speed = diff_speed
    model.split_times = model._calculate_split_times()
    model.transition_matrix = model._create_transition_matrix()
    
    # 运行完整模拟（300小时）
    model.sim_single(t_barcoding=132, t_collection=300)
    
    # 分析结果
    model.analyze_results(output_prefix="final_optimized_results")
    
    print("优化和验证完成!")

if __name__ == "__main__":
    main()