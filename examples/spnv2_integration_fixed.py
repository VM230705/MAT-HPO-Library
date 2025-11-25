"""
SPNV2 Integration Example for MAT-HPO Library

✅ 修復版本：展示如何正確使用修復後的 MAT-HPO Library

This example demonstrates how to integrate SPNV2 with the fixed MAT-HPO Library,
ensuring proper step tracking, consistent WandB logging, and val_f1_macro optimization.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional

# Add MAT_HPO_Library to path
sys.path.append('/home/vm230705/NTSC_Project_v2/MAT_HPO_Library')

from MAT_HPO_LIB.core.multi_agent_optimizer import MAT_HPO_Optimizer
from MAT_HPO_LIB.core.llm_enhanced_optimizer import LLMEnhancedMAT_HPO_Optimizer, LLMEnhancedOptimizationConfig
from MAT_HPO_LIB.core.evaluation_criteria import ModelSaveCriteria, OptimizationTarget, create_spnv2_criteria
from MAT_HPO_LIB.core.hyperparameter_space import HyperparameterSpace
from MAT_HPO_LIB.utils.config import OptimizationConfig
from MAT_HPO_LIB.utils.spnv2_config import SPNV2ConfigLoader, SPNV2HPOConfig
from MAT_HPO_LIB.utils.wandb_standards import WandBStandards

class SPNV2Environment:
    """
    SPNV2 環境適配器
    
    這個類別將 SPNV2 的訓練邏輯適配到 MAT-HPO Library 的環境接口
    """
    
    def __init__(self, dataset_name: str = "ICBEB", fold: int = 1):
        self.dataset_name = dataset_name
        self.fold = fold
        self.name = f"SPNV2-{dataset_name}-Fold{fold}"
        
        # 追蹤最佳結果
        self.best_val_f1 = float('-inf')
        self.best_step = -1
        self.best_hyperparams = None
        self.best_metrics = {}
        
    def reset(self) -> torch.Tensor:
        """重置環境狀態"""
        # 返回初始狀態張量
        return torch.zeros(1, 12)  # 9 class weights + 3 other params
    
    def step(self, hyperparams: Dict[str, Any]) -> tuple:
        """
        執行一步優化
        
        Args:
            hyperparams: 超參數字典
            
        Returns:
            tuple: (reward, metrics, done)
        """
        # 模擬 SPNV2 訓練過程
        # 在實際使用中，這裡會調用 SPNV2 的訓練代碼
        
        # 模擬訓練結果
        val_f1 = np.random.uniform(0.6, 0.8)  # 模擬 val_f1_macro
        val_acc = np.random.uniform(0.7, 0.9)
        test_f1 = np.random.uniform(0.6, 0.8)
        test_acc = np.random.uniform(0.7, 0.9)
        test_auc = np.random.uniform(0.8, 0.95)
        test_gmean = np.random.uniform(0.6, 0.8)
        
        # 創建指標字典
        metrics = {
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_precision': np.random.uniform(0.6, 0.8),
            'val_recall': np.random.uniform(0.6, 0.8),
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_precision': np.random.uniform(0.6, 0.8),
            'test_recall': np.random.uniform(0.6, 0.8),
            'test_auc': test_auc,
            'test_gmean': test_gmean
        }
        
        # 計算獎勵（基於 val_f1_macro）
        reward = val_f1
        
        # 檢查是否完成
        done = False
        
        return reward, metrics, done

def create_spnv2_hyperparameter_space() -> HyperparameterSpace:
    """
    創建 SPNV2 專用的超參數空間
    
    Returns:
        HyperparameterSpace: 配置好的超參數空間
    """
    space = HyperparameterSpace()
    
    # Agent 0: Class weights (9 classes)
    for i in range(9):
        space.add_continuous(f'class_weight_{i}', 0.1, 2.0, agent=0)
    
    # Agent 1: Architecture parameters
    space.add_continuous('hidden_size', 100, 500, agent=1)
    
    # Agent 2: Training parameters
    space.add_continuous('batch_size', 16, 64, agent=2)
    space.add_continuous('learning_rate', 1e-5, 1e-2, agent=2)
    
    return space

def run_spnv2_hpo_example():
    """
    運行 SPNV2 HPO 範例
    
    展示如何使用修復後的 MAT-HPO Library 進行 SPNV2 優化
    """
    print("🚀 開始 SPNV2 HPO 範例")
    print("=" * 50)
    
    # 1. 創建環境
    environment = SPNV2Environment(dataset_name="ICBEB", fold=1)
    print(f"✅ 環境創建完成: {environment.name}")
    
    # 2. 創建超參數空間
    hyperparameter_space = create_spnv2_hyperparameter_space()
    print(f"✅ 超參數空間創建完成: {len(hyperparameter_space.parameters)} 個參數")
    
    # 3. 創建評估標準（以 val_f1_macro 為主要目標）
    evaluation_criteria = create_spnv2_criteria()
    print(f"✅ 評估標準創建完成: {evaluation_criteria.primary_target.value}")
    
    # 4. 創建配置
    config = OptimizationConfig(
        max_steps=20,  # 較少的步驟用於範例
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    print(f"✅ 配置創建完成: {config.max_steps} 步驟")
    
    # 5. 創建優化器
    optimizer = MAT_HPO_Optimizer(
        environment=environment,
        hyperparameter_space=hyperparameter_space,
        config=config,
        evaluation_criteria=evaluation_criteria,
        output_dir="./spnv2_hpo_example_results"
    )
    print("✅ 優化器創建完成")
    
    # 6. 運行優化
    print("\n🔄 開始優化...")
    results = optimizer.optimize()
    
    # 7. 顯示結果
    print("\n📊 優化結果:")
    print(f"最佳步驟: {results['optimization_stats']['best_step']}")
    print(f"最佳 val_f1: {results['best_performance']['val_f1']:.4f}")
    print(f"最佳 test_f1: {results['best_performance']['test_f1']:.4f}")
    print(f"總時間: {results['optimization_stats']['total_time']:.2f} 秒")
    
    # 8. 驗證輸出檔案
    print("\n📁 輸出檔案檢查:")
    output_dir = "./spnv2_hpo_example_results"
    files_to_check = [
        'best_hyperparams.json',
        'optimization_results.json',
        'step_log.jsonl',
        'RL_model0.pt',
        'RL_model1.pt',
        'RL_model2.pt',
        'RL_model_input.pt',
        'CNNLSTM_model_hyp.npy'
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (缺失)")
    
    # 9. 檢查 best_hyperparams.json 內容
    best_hyp_path = os.path.join(output_dir, 'best_hyperparams.json')
    if os.path.exists(best_hyp_path):
        import json
        with open(best_hyp_path, 'r') as f:
            best_data = json.load(f)
        
        print(f"\n🔍 best_hyperparams.json 內容檢查:")
        print(f"  Step: {best_data.get('step', 'N/A')}")
        print(f"  Optimization Target: {best_data.get('optimization_target', 'N/A')}")
        print(f"  Primary Score: {best_data.get('performance', {}).get('primary_score', 'N/A')}")
        print(f"  Timestamp: {best_data.get('timestamp', 'N/A')}")
        
        # 驗證 step 是否正確
        if best_data.get('step') == results['optimization_stats']['best_step']:
            print("✅ Step 追蹤正確")
        else:
            print("❌ Step 追蹤錯誤")
    
    print("\n🎉 SPNV2 HPO 範例完成！")

def run_llm_enhanced_example():
    """
    運行 LLM Enhanced HPO 範例
    
    展示如何使用 LLM Enhanced 版本
    """
    print("\n🤖 開始 LLM Enhanced HPO 範例")
    print("=" * 50)
    
    # 1. 創建環境和超參數空間
    environment = SPNV2Environment(dataset_name="ICBEB", fold=2)
    hyperparameter_space = create_spnv2_hyperparameter_space()
    
    # 2. 創建 LLM Enhanced 配置
    llm_config = LLMEnhancedOptimizationConfig(
        max_steps=10,  # 較少的步驟用於範例
        enable_llm=False,  # 關閉 LLM 用於簡單範例
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    # 3. 創建 LLM Enhanced 優化器
    optimizer = LLMEnhancedMAT_HPO_Optimizer(
        environment=environment,
        hyperparameter_space=hyperparameter_space,
        config=llm_config,
        evaluation_criteria=create_spnv2_criteria(),
        output_dir="./spnv2_llm_hpo_example_results"
    )
    
    # 4. 運行優化
    print("🔄 開始 LLM Enhanced 優化...")
    results = optimizer.optimize()
    
    # 5. 顯示結果
    print("\n📊 LLM Enhanced 優化結果:")
    print(f"最佳步驟: {results['optimization_stats']['best_step']}")
    print(f"最佳 val_f1: {results['best_performance']['val_f1']:.4f}")
    
    print("\n🎉 LLM Enhanced HPO 範例完成！")

if __name__ == "__main__":
    # 運行基本範例
    run_spnv2_hpo_example()
    
    # 運行 LLM Enhanced 範例
    run_llm_enhanced_example()
    
    print("\n" + "=" * 50)
    print("🎯 所有範例完成！")
    print("📋 主要修復內容：")
    print("  ✅ 正確的 step 追蹤")
    print("  ✅ 統一的 WandB 記錄格式")
    print("  ✅ 以 val_f1_macro 為主要目標")
    print("  ✅ 靈活的評估標準")
    print("  ✅ 用戶可自訂的配置系統")
