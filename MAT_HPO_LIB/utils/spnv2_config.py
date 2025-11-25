"""
SPNV2 HPO Configuration System

This module provides flexible configuration management for SPNV2 HPO integration
with MAT_HPO_Library, allowing users to customize evaluation criteria, model saving
conditions, and optimization targets.
"""

import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class EvaluationConfig:
    """評估配置"""
    primary_target: str = "val_f1_macro"  # val_f1_macro, test_f1_macro, val_accuracy, custom
    secondary_targets: List[str] = None
    min_improvement_threshold: float = 0.001
    absolute_threshold: Optional[float] = None
    custom_evaluator_function: Optional[str] = None  # Python function as string
    
    def __post_init__(self):
        if self.secondary_targets is None:
            self.secondary_targets = ["test_f1_macro"]

@dataclass
class ModelSavingConfig:
    """模型保存配置"""
    save_on_improvement: bool = True
    save_on_milestone: bool = True
    milestone_steps: List[int] = None
    save_checkpoint_every: int = 10  # Save checkpoint every N steps
    max_checkpoints: int = 5  # Maximum number of checkpoints to keep
    
    def __post_init__(self):
        if self.milestone_steps is None:
            self.milestone_steps = [25, 50, 75, 100, 125, 150, 175, 200]

@dataclass
class WandBConfig:
    """WandB 配置"""
    unified_format: bool = True
    log_hyperparams: bool = True
    log_best_metrics: bool = True
    log_step_details: bool = True
    log_consistency_checks: bool = True
    project_name: str = "SPNV2-HPO"
    entity: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["hpo", "spnv2", "mat-hpo"]

@dataclass
class MATHPOConfig:
    """MAT-HPO 配置"""
    max_steps: int = 50
    early_stopping_patience: int = 10
    device: str = "cuda:0"
    batch_size: int = 32
    learning_rate: float = 0.001
    behaviour_update_freq: int = 1
    gradient_clip: float = 1.0
    replay_buffer_size: int = 10000
    target_update_freq: int = 10
    exploration_noise: float = 0.1

@dataclass
class SPNV2HPOConfig:
    """SPNV2 HPO 完整配置"""
    evaluation: EvaluationConfig
    model_saving: ModelSavingConfig
    wandb: WandBConfig
    mat_hpo: MATHPOConfig
    
    def __init__(self, **kwargs):
        # 從 kwargs 中提取各部分的配置
        self.evaluation = EvaluationConfig(**kwargs.get('evaluation', {}))
        self.model_saving = ModelSavingConfig(**kwargs.get('model_saving', {}))
        self.wandb = WandBConfig(**kwargs.get('wandb', {}))
        self.mat_hpo = MATHPOConfig(**kwargs.get('mat_hpo', {}))

class SPNV2ConfigLoader:
    """SPNV2 配置載入器"""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> SPNV2HPOConfig:
        """
        從 YAML 檔案載入配置
        
        Args:
            config_path: YAML 配置檔案路徑
            
        Returns:
            SPNV2HPOConfig 物件
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return SPNV2HPOConfig(**config_dict)
    
    @staticmethod
    def load_from_json(config_path: str) -> SPNV2HPOConfig:
        """
        從 JSON 檔案載入配置
        
        Args:
            config_path: JSON 配置檔案路徑
            
        Returns:
            SPNV2HPOConfig 物件
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return SPNV2HPOConfig(**config_dict)
    
    @staticmethod
    def save_to_yaml(config: SPNV2HPOConfig, output_path: str):
        """
        保存配置到 YAML 檔案
        
        Args:
            config: SPNV2HPOConfig 物件
            output_path: 輸出檔案路徑
        """
        config_dict = asdict(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def save_to_json(config: SPNV2HPOConfig, output_path: str):
        """
        保存配置到 JSON 檔案
        
        Args:
            config: SPNV2HPOConfig 物件
            output_path: 輸出檔案路徑
        """
        config_dict = asdict(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_default_config() -> SPNV2HPOConfig:
        """
        創建預設配置
        
        Returns:
            預設的 SPNV2HPOConfig 物件
        """
        return SPNV2HPOConfig()
    
    @staticmethod
    def create_spnv2_optimized_config() -> SPNV2HPOConfig:
        """
        創建針對 SPNV2 優化的配置
        
        Returns:
            針對 SPNV2 優化的配置
        """
        return SPNV2HPOConfig(
            evaluation={
                'primary_target': 'val_f1_macro',
                'secondary_targets': ['test_f1_macro'],
                'min_improvement_threshold': 0.001,
                'absolute_threshold': None
            },
            model_saving={
                'save_on_improvement': True,
                'save_on_milestone': True,
                'milestone_steps': [25, 50, 75, 100, 125, 150, 175, 200],
                'save_checkpoint_every': 10,
                'max_checkpoints': 5
            },
            wandb={
                'unified_format': True,
                'log_hyperparams': True,
                'log_best_metrics': True,
                'log_step_details': True,
                'log_consistency_checks': True,
                'project_name': 'SPNV2-HPO',
                'tags': ['hpo', 'spnv2', 'mat-hpo', 'val_f1_macro']
            },
            mat_hpo={
                'max_steps': 50,
                'early_stopping_patience': 10,
                'device': 'cuda:0',
                'batch_size': 32,
                'learning_rate': 0.001,
                'behaviour_update_freq': 1,
                'gradient_clip': 1.0,
                'replay_buffer_size': 10000,
                'target_update_freq': 10,
                'exploration_noise': 0.1
            }
        )

def create_default_yaml_config(output_path: str = "spnv2_hpo_config.yaml"):
    """
    創建預設的 YAML 配置檔案
    
    Args:
        output_path: 輸出檔案路徑
    """
    config = SPNV2ConfigLoader.create_spnv2_optimized_config()
    SPNV2ConfigLoader.save_to_yaml(config, output_path)
    print(f"✅ 預設配置已保存到: {output_path}")

def create_default_json_config(output_path: str = "spnv2_hpo_config.json"):
    """
    創建預設的 JSON 配置檔案
    
    Args:
        output_path: 輸出檔案路徑
    """
    config = SPNV2ConfigLoader.create_spnv2_optimized_config()
    SPNV2ConfigLoader.save_to_json(config, output_path)
    print(f"✅ 預設配置已保存到: {output_path}")

# 預設配置範例
DEFAULT_CONFIG_YAML = """
# SPNV2 HPO Configuration
# 針對 SPNV2 優化的 MAT-HPO Library 配置

evaluation:
  primary_target: "val_f1_macro"  # 主要優化目標
  secondary_targets: ["test_f1_macro"]  # 次要目標
  min_improvement_threshold: 0.001  # 最小改進閾值
  absolute_threshold: null  # 絕對閾值（可選）

model_saving:
  save_on_improvement: true  # 改進時保存
  save_on_milestone: true  # 里程碑時保存
  milestone_steps: [25, 50, 75, 100, 125, 150, 175, 200]  # 里程碑步驟
  save_checkpoint_every: 10  # 每 N 步保存檢查點
  max_checkpoints: 5  # 最大檢查點數量

wandb:
  unified_format: true  # 統一格式
  log_hyperparams: true  # 記錄超參數
  log_best_metrics: true  # 記錄最佳指標
  log_step_details: true  # 記錄步驟詳情
  log_consistency_checks: true  # 記錄一致性檢查
  project_name: "SPNV2-HPO"  # WandB 專案名稱
  entity: null  # WandB 實體（可選）
  tags: ["hpo", "spnv2", "mat-hpo", "val_f1_macro"]  # 標籤

mat_hpo:
  max_steps: 50  # 最大步驟數
  early_stopping_patience: 10  # 早停耐心
  device: "cuda:0"  # 設備
  batch_size: 32  # 批次大小
  learning_rate: 0.001  # 學習率
  behaviour_update_freq: 1  # 行為更新頻率
  gradient_clip: 1.0  # 梯度裁剪
  replay_buffer_size: 10000  # 重播緩衝區大小
  target_update_freq: 10  # 目標更新頻率
  exploration_noise: 0.1  # 探索噪聲
"""

if __name__ == "__main__":
    # 創建預設配置檔案
    create_default_yaml_config("spnv2_hpo_config.yaml")
    create_default_json_config("spnv2_hpo_config.json")
    
    print("📋 配置檔案已創建！")
    print("📝 您可以編輯這些檔案來自訂 HPO 行為")
    print("🔧 主要可自訂項目：")
    print("   - evaluation.primary_target: 主要優化目標")
    print("   - model_saving.milestone_steps: 保存里程碑")
    print("   - wandb.project_name: WandB 專案名稱")
    print("   - mat_hpo.max_steps: 最大優化步驟")
