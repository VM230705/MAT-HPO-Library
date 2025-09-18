#!/usr/bin/env python3
"""
Configurable MAT-HPO Optimizer - 分層級超參數優化器

專業級多智能體強化學習超參數優化，支援LLM增強和分層級配置：
1. 分層級參數設定 - 按需顯示相關參數
2. 智能默認值 - 基於最佳實踐的自動配置
3. 漸進式複雜度 - 從簡單到專業級精確控制
4. LLM增強優化 - 支援多種智能策略
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import warnings

from .core.llm_enhanced_optimizer import LLMEnhancedMAT_HPO_Optimizer, LLMEnhancedOptimizationConfig
from .core.enhanced_environment import TimeSeriesEnvironment
from .core.hyperparameter_space import HyperparameterSpace
from .easy_hpo import EasyHPOEnvironment


class LLMEnhancedHPO:
    """
    LLM增強超參數優化器 (LLM Enhanced Hyperparameter Optimizer)

    基於多智能體強化學習的超參數優化器，可選LLM增強，提供分層級配置：
    - 核心層：任務類型和試驗次數（必需）
    - LLM層：LLM增強策略（可選）
    - 配置層：LLM和RL詳細參數（可選）
    """

    def __init__(self,
                 # === 核心層級 (必需) ===
                 task_type: str = "time_series_classification",
                 max_trials: int = 30,

                 # === LLM層級 (可選) ===
                 llm_enabled: bool = False,
                 llm_strategy: str = "adaptive",  # "fixed_alpha", "adaptive"

                 # === 配置層級 (可選) ===
                 llm_config: Optional[Dict[str, Any]] = None,

                 # === 通用設定 ===
                 seed: int = 42,
                 verbose: bool = True,
                 output_dir: str = "./llm_enhanced_hpo_results"):
        """
        初始化LLM增強HPO優化器

        Args:
            # 核心參數
            task_type: 任務類型 ("time_series_classification", "ecg_classification", "classification")
            max_trials: 最大試驗次數

            # LLM智能層參數 (僅當llm_enabled=True時生效)
            llm_enabled: 是否啟用LLM智能增強
            llm_strategy: LLM策略
                - "fixed_alpha": 固定混合比例 (alpha參數控制LLM/RL比例)
                - "adaptive": 監控RL性能斜率的自適應觸發 (基於論文 arXiv:2507.13712)

            # 專家配置 (字典形式，可選)
            expert_config: 專家級參數字典，可包含：
                # LLM特定參數
                - 'alpha': 固定混合比例，僅用於fixed_alpha策略 (default: 0.3)
                - 'performance_threshold': RL性能斜率閾值，用於adaptive策略 (default: 0.01)
                - 'llm_model': LLM模型 (default: "llama3.2:3b")
                - 'llm_cooldown': LLM冷卻期 (default: 5)

                # RL特定參數
                - 'buffer_size': 經驗回放大小 (default: 1000)
                - 'learning_rate': RL學習率 (default: 0.001)
                - 'gamma': 折扣因子 (default: 0.99)
                - 'tau': 軟更新係數 (default: 0.001)

                # 其他參數
                - 'device': 計算設備 (default: "auto")

            # 通用設定
            seed: 隨機種子
            verbose: 顯示詳細信息
            output_dir: 輸出目錄
        """

        # === 核心參數 ===
        self.task_type = task_type
        self.max_trials = max_trials
        self.seed = seed
        self.verbose = verbose
        self.output_dir = Path(output_dir)

        # === LLM參數處理 ===
        self.llm_enabled = llm_enabled
        self.llm_strategy = llm_strategy if llm_enabled else None

        # === LLM配置處理 ===
        self.llm_config = llm_config or {}

        # === 配置默認值 ===
        self._setup_default_configs()

        # === 驗證配置 ===
        self._validate_configuration()

        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化組件
        self.environment = None
        self.hyperparameter_space = None
        self.optimizer = None
        self.results = None

        # 顯示配置摘要
        self._display_configuration()

    def _setup_default_configs(self):
        """設定配置默認值"""

        # LLM默認值
        if self.llm_enabled:
            llm_defaults = {
                'model': 'llama3.2:3b',
                'base_url': 'http://localhost:11434',
                'alpha': 0.3,
                'slope_threshold': 0.01,
                'cooldown': 5,
                'min_episodes': 5
            }
            # 合併用戶配置和默認值
            self.llm_config = {**llm_defaults, **self.llm_config}
        else:
            self.llm_config = None

        # RL默認值 (根據任務類型調整)
        if "time_series" in self.task_type or "ecg" in self.task_type:
            # 時間序列任務的優化默認值
            rl_defaults = {
                'buffer_size': 1000,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'tau': 0.001,
                'hidden_dim': 256,
                'batch_size': 64
            }
        else:
            # 通用分類任務的默認值
            rl_defaults = {
                'buffer_size': 800,
                'learning_rate': 0.002,
                'gamma': 0.95,
                'tau': 0.002,
                'hidden_dim': 128,
                'batch_size': 32
            }

        # 應用用戶自定義的RL參數
        if self.llm_config:
            rl_overrides = {k: v for k, v in self.llm_config.items()
                           if k in rl_defaults and v is not None}
            self.rl_config = {**rl_defaults, **rl_overrides}
            # 設備設定
            self.device = self.llm_config.get('device', 'auto')
        else:
            self.rl_config = rl_defaults
            self.device = 'auto'

    def _validate_configuration(self):
        """驗證配置的有效性"""

        # 驗證任務類型
        valid_tasks = ["time_series_classification", "ecg_classification", "classification", "image_classification"]
        if self.task_type not in valid_tasks:
            warnings.warn(f"Unknown task_type '{self.task_type}', using default hyperparameter space")

        # 驗證LLM策略
        if self.llm_enabled:
            valid_strategies = ["fixed_alpha", "adaptive"]
            if self.llm_strategy not in valid_strategies:
                raise ValueError(f"Invalid llm_strategy '{self.llm_strategy}'. Must be one of {valid_strategies}")

        # 驗證數值範圍
        if self.llm_enabled and self.llm_config:
            if not (0 <= self.llm_config['alpha'] <= 1):
                raise ValueError(f"alpha must be between 0 and 1, got {self.llm_config['alpha']}")
            if self.llm_config['performance_threshold'] <= 0:
                raise ValueError(f"performance_threshold must be positive, got {self.llm_config['performance_threshold']}")

    def _display_configuration(self):
        """顯示當前配置摘要"""
        if not self.verbose:
            return

        print(f"🚀 Intelligent HPO Configuration:")
        print(f"   Task: {self.task_type}")
        print(f"   Max trials: {self.max_trials}")

        if self.llm_enabled:
            print(f"   🤖 LLM: ✅ Enabled ({self.llm_strategy})")
            if self.llm_strategy == "fixed_alpha":
                print(f"      Alpha: {self.llm_config['alpha']} ({self.llm_config['alpha']*100:.0f}% LLM + {(1-self.llm_config['alpha'])*100:.0f}% RL)")
            elif self.llm_strategy == "adaptive":
                print(f"      Performance threshold: {self.llm_config['performance_threshold']}")
            print(f"      Model: {self.llm_config['model']}")
        else:
            print(f"   🤖 LLM: ❌ Disabled (Pure RL)")

        print(f"   🧠 RL: buffer={self.rl_config['buffer_size']}, lr={self.rl_config['learning_rate']}")
        print(f"   📁 Output: {self.output_dir}")

    def optimize(self,
                 X_train, y_train,
                 X_val=None, y_val=None,
                 X_test=None, y_test=None,
                 custom_space: Optional[HyperparameterSpace] = None) -> Dict[str, Any]:
        """
        運行優化

        Args:
            X_train, y_train: 訓練數據
            X_val, y_val: 驗證數據 (可選，會自動分割)
            X_test, y_test: 測試數據 (可選)
            custom_space: 自定義超參數空間 (可選)

        Returns:
            優化結果字典
        """

        # 自動分割驗證集
        if X_val is None and y_val is None:
            from sklearn.model_selection import train_test_split
            import numpy as np
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.seed,
                stratify=y_train if len(np.unique(y_train)) > 1 else None
            )
            if self.verbose:
                print(f"📊 Auto-split: train={len(X_train)}, val={len(X_val)}")

        # 設置環境 - 使用已經工作的EasyHPOEnvironment
        self.environment = EasyHPOEnvironment(
            task_type=self.task_type,
            dataset_name=f"ConfigurableMAT_HPO_{self.task_type}"
        )
        self.environment.set_data(X_train, y_train, X_val, y_val, X_test, y_test)
        self.environment.load_data()

        # 設置超參數空間
        if custom_space:
            self.hyperparameter_space = custom_space
        else:
            self.hyperparameter_space = self._create_smart_space()

        # 創建優化配置
        config = self._create_optimization_config()

        # 創建優化器
        self.optimizer = LLMEnhancedMAT_HPO_Optimizer(
            environment=self.environment,
            hyperparameter_space=self.hyperparameter_space,
            config=config
        )

        if self.verbose:
            print(f"\\n⚡ Starting optimization...")

        # 運行優化
        import time
        start_time = time.time()
        self.results = self.optimizer.optimize()
        end_time = time.time()

        # 添加統計信息
        self.results['optimization_time'] = end_time - start_time
        self.results['trials_completed'] = self.max_trials
        self.results['configuration'] = self._get_config_summary()

        if self.verbose:
            self._display_results()

        return self.results

    def _create_smart_space(self) -> HyperparameterSpace:
        """根據任務類型創建智能超參數空間，確保每個agent都有參數"""
        space = HyperparameterSpace()

        if "time_series" in self.task_type or "ecg" in self.task_type:
            # 時間序列優化空間
            space.add_discrete("hidden_size", [64, 128, 256, 512], agent=0)
            space.add_continuous("dropout", 0.0, 0.5, agent=0)

            space.add_continuous("learning_rate", 0.0001, 0.01, agent=1)
            space.add_discrete("batch_size", [16, 32, 64, 128], agent=1)

            # 確保agent 2有參數
            space.add_continuous("regularization", 1e-6, 1e-3, agent=2)

            # ECG特定的類別權重
            if "ecg" in self.task_type and hasattr(self.environment, 'y_train'):
                import numpy as np
                num_classes = len(np.unique(self.environment.y_train))
                if num_classes > 2:
                    for i in range(min(num_classes, 9)):  # 最多9個類別權重
                        space.add_continuous(f"class_weight_{i}", 0.5, 3.0, agent=2)

        elif "image" in self.task_type:
            # 圖像分類空間
            space.add_discrete("filters", [32, 64, 128, 256], agent=0)
            space.add_discrete("kernel_size", [3, 5, 7], agent=0)

            space.add_continuous("learning_rate", 0.0001, 0.001, agent=1)
            space.add_discrete("batch_size", [32, 64, 128], agent=1)

            space.add_continuous("regularization", 1e-6, 1e-3, agent=2)

        else:
            # 通用分類空間
            space.add_continuous("learning_rate", 0.0001, 0.1, agent=0)
            space.add_continuous("regularization", 1e-6, 1e-2, agent=0)

            space.add_discrete("batch_size", [16, 32, 64, 128], agent=1)
            space.add_continuous("momentum", 0.8, 0.99, agent=1)

            space.add_continuous("weight_decay", 1e-6, 1e-3, agent=2)
            space.add_discrete("optimizer_type", ['adam', 'sgd'], agent=2)

        return space

    def _create_optimization_config(self) -> LLMEnhancedOptimizationConfig:
        """創建優化配置"""

        # 使用類似EasyHPO的簡化配置方式
        config_params = {
            'max_steps': self.max_trials,
            'replay_buffer_size': self.rl_config['buffer_size'],
            'verbose': self.verbose,
            'dataset_name': self.environment.dataset_name,
            'task_description': self.environment.get_llm_context()
        }

        # LLM參數 (僅當啟用時添加)
        if self.llm_enabled:
            config_params.update({
                'enable_llm': True,
                'llm_model': self.llm_config['model'],
                'llm_base_url': self.llm_config['base_url'],
                'mixing_strategy': self.llm_strategy,
                'alpha': self.llm_config['alpha'],
                'slope_threshold': self.llm_config['slope_threshold'],
                'llm_cooldown_episodes': self.llm_config['cooldown'],
                'min_episodes_before_llm': self.llm_config['min_episodes']
            })
        else:
            config_params['enable_llm'] = False

        return LLMEnhancedOptimizationConfig(**config_params)

    def _get_config_summary(self) -> Dict[str, Any]:
        """獲取配置摘要"""
        summary = {
            'task_type': self.task_type,
            'max_trials': self.max_trials,
            'llm_enabled': self.llm_enabled,
            'rl_config': self.rl_config
        }

        if self.llm_enabled:
            summary['llm_config'] = {
                'strategy': self.llm_strategy,
                'model': self.llm_config['model'],
                'alpha': self.llm_config['alpha'],
                'slope_threshold': self.llm_config['slope_threshold']
            }

        return summary

    def _display_results(self):
        """顯示優化結果"""
        if not self.results:
            return

        print(f"\\n🎉 Optimization Complete!")
        print(f"   Time: {self.results['optimization_time']:.1f} seconds")

        if 'best_performance' in self.results:
            print(f"\\n🏆 Best Performance:")
            for metric, value in self.results['best_performance'].items():
                print(f"   {metric}: {value:.4f}")

        if 'best_hyperparameters' in self.results:
            print(f"\\n⚙️  Best Hyperparameters:")
            for param, value in self.results['best_hyperparameters'].items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.6f}")
                else:
                    print(f"   {param}: {value}")

    def get_best_config(self) -> Dict[str, Any]:
        """獲取最佳配置"""
        if self.results and 'best_hyperparameters' in self.results:
            return self.results['best_hyperparameters']
        return {}

    def get_config_summary(self) -> Dict[str, Any]:
        """獲取配置摘要"""
        if self.results and 'configuration' in self.results:
            return self.results['configuration']
        return self._get_config_summary()


# === 便利函數 ===

def optimize_with_llm(X_train, y_train, X_val=None, y_val=None,
                     task_type="time_series_classification",
                     strategy="adaptive",
                     max_trials=30,
                     **expert_config):
    """
    LLM增強的一行式優化

    Args:
        X_train, y_train: 訓練數據
        X_val, y_val: 驗證數據
        task_type: 任務類型
        strategy: LLM策略 ("fixed_alpha", "adaptive")
        max_trials: 最大試驗次數
        **expert_config: 專家配置參數

    Returns:
        最佳超參數配置字典
    """
    optimizer = IntelligentHPO(
        task_type=task_type,
        max_trials=max_trials,
        llm_enabled=True,
        llm_strategy=strategy,
        expert_config=expert_config
    )
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    return optimizer.get_best_config()


def optimize_pure_rl(X_train, y_train, X_val=None, y_val=None,
                    task_type="time_series_classification",
                    max_trials=30,
                    **advanced_config):
    """
    純多智能體強化學習優化

    Args:
        X_train, y_train: 訓練數據
        X_val, y_val: 驗證數據
        task_type: 任務類型
        max_trials: 最大試驗次數
        **expert_config: 專家配置參數

    Returns:
        最佳超參數配置字典
    """
    optimizer = IntelligentHPO(
        task_type=task_type,
        max_trials=max_trials,
        llm_enabled=False,
        expert_config=expert_config
    )
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    return optimizer.get_best_config()