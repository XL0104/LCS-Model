"""
消融实验：测试四个关键模块对链接预测性能的影响
模块1: 子图提取模块
模块2: 特征表示模块  
模块3: 特征融合模块
模块4: 动态图卷积预测模块
"""

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from seal_link_pred import run_sweal, SWEALArgumentParser
from utils import extract_enclosing_subgraphs, k_hop_subgraph
from models import DGCNN, MVGCN
import json


class AblationStudy:
    def __init__(self, base_args, device):
        self.base_args = base_args
        self.device = device
        self.results = {}
        
    def run_baseline(self):
        """运行完整模型作为基线"""
        print("=== 运行完整模型（基线）===")
        baseline_results = run_sweal(self.base_args, self.device)
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def ablation_no_subgraph_extraction(self):
        """消融实验1：移除子图提取模块"""
        print("=== 消融实验1：移除子图提取模块 ===")
        
        # 修改参数：使用0跳子图（即不使用子图提取）
        args = self._copy_args()
        args.num_hops = 0  # 不使用子图提取
        args.save_appendix = '_ablation_no_subgraph'
        
        try:
            results = run_sweal(args, self.device)
            self.results['no_subgraph'] = results
            return results
        except Exception as e:
            print(f"消融实验1失败: {e}")
            return None
    
    def ablation_no_feature_representation(self):
        """消融实验2：移除特征表示模块"""
        print("=== 消融实验2：移除特征表示模块 ===")
        
        # 修改参数：不使用节点特征
        args = self._copy_args()
        args.use_feature = False  # 不使用节点特征
        args.save_appendix = '_ablation_no_features'
        
        try:
            results = run_sweal(args, self.device)
            self.results['no_features'] = results
            return results
        except Exception as e:
            print(f"消融实验2失败: {e}")
            return None
    
    def ablation_no_feature_fusion(self):
        """消融实验3：移除特征融合模块"""
        print("=== 消融实验3：移除特征融合模块 ===")
        
        # 修改参数：使用单视图DGCNN而不是多视图MVGCN
        args = self._copy_args()
        args.model = 'DGCNN'  # 使用单视图模型
        args.save_appendix = '_ablation_no_fusion'
        
        try:
            results = run_sweal(args, self.device)
            self.results['no_fusion'] = results
            return results
        except Exception as e:
            print(f"消融实验3失败: {e}")
            return None
    
    def ablation_no_dynamic_gcn(self):
        """消融实验4：移除动态图卷积预测模块"""
        print("=== 消融实验4：移除动态图卷积预测模块 ===")
        
        # 修改参数：使用简单的MLP而不是DGCNN
        args = self._copy_args()
        args.model = 'MLP'  # 使用简单MLP模型
        args.save_appendix = '_ablation_no_dgcnn'
        
        try:
            results = run_sweal(args, self.device)
            self.results['no_dgcnn'] = results
            return results
        except Exception as e:
            print(f"消融实验4失败: {e}")
            return None
    
    def cross_ablation_study(self):
        """交叉消融实验：组合移除多个模块"""
        print("=== 交叉消融实验 ===")
        
        cross_results = {}
        
        # 组合1：移除子图提取和特征表示
        print("交叉实验1：移除子图提取+特征表示")
        args = self._copy_args()
        args.num_hops = 0
        args.use_feature = False
        args.save_appendix = '_cross_no_subgraph_features'
        try:
            results = run_sweal(args, self.device)
            cross_results['no_subgraph_features'] = results
        except Exception as e:
            print(f"交叉实验1失败: {e}")
        
        # 组合2：移除特征表示和特征融合
        print("交叉实验2：移除特征表示+特征融合")
        args = self._copy_args()
        args.use_feature = False
        args.model = 'DGCNN'
        args.save_appendix = '_cross_no_features_fusion'
        try:
            results = run_sweal(args, self.device)
            cross_results['no_features_fusion'] = results
        except Exception as e:
            print(f"交叉实验2失败: {e}")
        
        # 组合3：移除特征融合和动态图卷积
        print("交叉实验3：移除特征融合+动态图卷积")
        args = self._copy_args()
        args.model = 'MLP'
        args.save_appendix = '_cross_no_fusion_dgcnn'
        try:
            results = run_sweal(args, self.device)
            cross_results['no_fusion_dgcnn'] = results
        except Exception as e:
            print(f"交叉实验3失败: {e}")
        
        # 组合4：移除所有模块（最简模型）
        print("交叉实验4：移除所有模块")
        args = self._copy_args()
        args.num_hops = 0
        args.use_feature = False
        args.model = 'MLP'
        args.save_appendix = '_cross_no_all'
        try:
            results = run_sweal(args, self.device)
            cross_results['no_all'] = results
        except Exception as e:
            print(f"交叉实验4失败: {e}")
        
        self.results['cross'] = cross_results
        return cross_results
    
    def _copy_args(self):
        """复制基础参数"""
        # 创建一个新的参数对象，复制所有属性
        args_copy = SWEALArgumentParser(
            dataset=self.base_args.dataset,
            fast_split=self.base_args.fast_split,
            model=self.base_args.model,
            sortpool_k=self.base_args.sortpool_k,
            num_layers=self.base_args.num_layers,
            hidden_channels=self.base_args.hidden_channels,
            batch_size=self.base_args.batch_size,
            num_hops=self.base_args.num_hops,
            ratio_per_hop=self.base_args.ratio_per_hop,
            max_nodes_per_hop=self.base_args.max_nodes_per_hop,
            node_label=self.base_args.node_label,
            use_feature=self.base_args.use_feature,
            use_edge_weight=self.base_args.use_edge_weight,
            lr=self.base_args.lr,
            epochs=self.base_args.epochs,
            runs=self.base_args.runs,
            train_percent=self.base_args.train_percent,
            val_percent=self.base_args.val_percent,
            test_percent=self.base_args.test_percent,
            dynamic_train=self.base_args.dynamic_train,
            dynamic_val=self.base_args.dynamic_val,
            dynamic_test=self.base_args.dynamic_test,
            num_workers=self.base_args.num_workers,
            train_node_embedding=self.base_args.train_node_embedding,
            pretrained_node_embedding=self.base_args.pretrained_node_embedding,
            use_valedges_as_input=self.base_args.use_valedges_as_input,
            eval_steps=self.base_args.eval_steps,
            log_steps=self.base_args.log_steps,
            data_appendix=self.base_args.data_appendix,
            save_appendix=self.base_args.save_appendix,
            keep_old=self.base_args.keep_old,
            continue_from=self.base_args.continue_from,
            only_test=self.base_args.only_test,
            test_multiple_models=self.base_args.test_multiple_models,
            use_heuristic=self.base_args.use_heuristic,
            m=self.base_args.m,
            M=self.base_args.M,
            dropedge=self.base_args.dropedge,
            calc_ratio=self.base_args.calc_ratio,
            checkpoint_training=self.base_args.checkpoint_training,
            delete_dataset=self.base_args.delete_dataset,
            pairwise=self.base_args.pairwise,
            loss_fn=self.base_args.loss_fn,
            neg_ratio=self.base_args.neg_ratio,
            profile=self.base_args.profile,
            split_val_ratio=self.base_args.split_val_ratio,
            split_test_ratio=self.base_args.split_test_ratio,
            train_mlp=self.base_args.train_mlp,
            dropout=self.base_args.dropout,
            train_gae=self.base_args.train_gae,
            base_gae=self.base_args.base_gae,
            dataset_stats=self.base_args.dataset_stats,
            seed=self.base_args.seed,
            dataset_split_num=self.base_args.dataset_split_num,
            train_n2v=self.base_args.train_n2v,
            train_mf=self.base_args.train_mf
        )
        return args_copy
    
    def save_results(self, filename='ablation_results.json'):
        """保存消融实验结果"""
        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in self.results.items():
            if value is not None:
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)) or subvalue is None:
                            serializable_results[key][subkey] = subvalue
                        elif isinstance(subvalue, (list, tuple)):
                            serializable_results[key][subkey] = list(subvalue)
                        else:
                            serializable_results[key][subkey] = str(subvalue)
                else:
                    serializable_results[key] = str(value)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到 {filename}")
    
    def print_summary(self):
        """打印消融实验摘要"""
        print("\n" + "="*80)
        print("消融实验摘要")
        print("="*80)
        
        metrics = ['AUC', 'AP']
        
        for metric in metrics:
            print(f"\n{metric} 性能对比:")
            print("-" * 40)
            
            for exp_name, results in self.results.items():
                if results and 'test' in results and metric in results['test']:
                    test_value = results['test'][metric]
                    if isinstance(test_value, (list, tuple)) and len(test_value) >= 2:
                        test_value = test_value[1]  # 取测试集结果
                    print(f"{exp_name:25}: {test_value:.4f}")
        
        # 计算性能下降百分比
        if 'baseline' in self.results and self.results['baseline']:
            baseline_auc = self.results['baseline']['test']['AUC'][1]
            baseline_ap = self.results['baseline']['test']['AP'][1]
            
            print("\n性能下降百分比 (相对于基线):")
            print("-" * 40)
            
            for exp_name, results in self.results.items():
                if exp_name != 'baseline' and results and 'test' in results:
                    test_auc = results['test']['AUC'][1] if 'AUC' in results['test'] else 0
                    test_ap = results['test']['AP'][1] if 'AP' in results['test'] else 0
                    
                    auc_drop = ((baseline_auc - test_auc) / baseline_auc) * 100
                    ap_drop = ((baseline_ap - test_ap) / baseline_ap) * 100
                    
                    print(f"{exp_name:25}: AUC下降 {auc_drop:6.2f}%, AP下降 {ap_drop:6.2f}%")


def main():
    """主函数：运行消融实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 基础参数设置（与seal_link_pred.py相同的数据集和配置）
    base_args = SWEALArgumentParser(
        dataset='Cora',  # 使用Cora数据集
        fast_split=False,  # 修复：fast_split功能未实现，设为False
        model='MVGCN',  # 使用多视图GCN作为基线
        sortpool_k=0.6,
        num_layers=3,
        hidden_channels=32,
        batch_size=32,
        num_hops=2,  # 使用2跳子图
        ratio_per_hop=1.0,
        max_nodes_per_hop=None,
        node_label='drnl',
        use_feature=True,  # 使用节点特征
        use_edge_weight=False,
        lr=0.001,
        epochs=50,
        runs=3,  # 多次运行取平均
        train_percent=100,
        val_percent=10,
        test_percent=10,
        dynamic_train=False,
        dynamic_val=False,
        dynamic_test=False,
        num_workers=4,
        train_node_embedding=False,
        pretrained_node_embedding=None,
        use_valedges_as_input=False,
        eval_steps=1,
        log_steps=1,
        data_appendix='',
        save_appendix='',
        keep_old=False,
        continue_from=None,
        only_test=False,
        test_multiple_models=False,
        use_heuristic=False,
        m=None,
        M=None,
        dropedge=0.0,
        calc_ratio=False,
        checkpoint_training=False,
        delete_dataset=False,
        pairwise=False,
        loss_fn='auc',
        neg_ratio=1,
        profile=False,
        split_val_ratio=0.05,
        split_test_ratio=0.1,
        train_mlp=False,
        dropout=0.5,
        train_gae=False,
        base_gae=False,
        dataset_stats=False,
        seed=42,
        dataset_split_num=0,
        train_n2v=False,
        train_mf=False
    )
    
    # 创建消融实验对象
    study = AblationStudy(base_args, device)
    
    # 运行基线实验
    study.run_baseline()
    
    # 运行单个模块消融实验
    study.ablation_no_subgraph_extraction()
    study.ablation_no_feature_representation()
    study.ablation_no_feature_fusion()
    study.ablation_no_dynamic_gcn()
    
    # 运行交叉消融实验
    study.cross_ablation_study()
    
    # 保存和显示结果
    study.save_results('ablation_results.json')
    study.print_summary()


if __name__ == '__main__':
    main()