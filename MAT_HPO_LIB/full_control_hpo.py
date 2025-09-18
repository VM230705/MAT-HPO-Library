#!/usr/bin/env python3
"""
Full Control HPO - 完全參數化的MAT-HPO優化器
保留所有核心功能，允許用戶控制每個參數
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .core.llm_enhanced_optimizer import LLMEnhancedMAT_HPO_Optimizer, LLMEnhancedOptimizationConfig
from .core.enhanced_environment import TimeSeriesEnvironment
from .core.hyperparameter_space import HyperparameterSpace
from .utils.llm_config import create_llm_config


class FullControlHPO:
    """
    完全控制的HPO接口 - 暴露所有MAT-HPO + LLM參數

    這個類不隱藏任何參數，讓用戶完全控制優化流程
    """

    def __init__(self,
                 # 基本配置
                 task_type: str = "time_series_classification",
                 max_trials: int = 30,

                 # LLM 配置
                 llm_enabled: bool = True,
                 llm_model: str = "llama3.2:3b",
                 llm_base_url: str = "http://localhost:11434",
                 mixing_strategy: str = "fixed_alpha",

                 # LLM 混合參數 - 完全暴露
                 alpha: float = 0.3,                    # 固定混合比例
                 slope_threshold: float = 0.01,         # 線性回歸斜率閾值
                 llm_cooldown_episodes: int = 5,        # LLM 冷卻期
                 min_episodes_before_llm: int = 5,      # 最小episode後才能用LLM

                 # RL 配置 - 完全暴露
                 replay_buffer_size: int = 1000,        # 經驗回放緩衝區大小
                 gamma: float = 0.99,                   # 折扣因子
                 tau: float = 0.001,                    # 軟更新係數
                 batch_size: int = 64,                  # RL批次大小
                 learning_rate: float = 0.001,          # RL學習率
                 actor_lr: float = 0.001,              # Actor學習率
                 critic_lr: float = 0.002,             # Critic學習率
                 hidden_dim: int = 256,                # 隱藏層維度

                 # 其他高級參數
                 seed: int = 42,                       # 隨機種子
                 device: str = "auto",                 # 設備選擇
                 verbose: bool = True,
                 output_dir: str = "./full_control_results",

                 # 數據集參數
                 dataset_info_csv_path: Optional[str] = None,
                 custom_prompt_template: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 task_description: Optional[str] = None,

                 **kwargs):
        """
        初始化完全控制HPO優化器

        Args:
            task_type: 任務類型
            max_trials: 最大試驗次數

            # LLM 參數
            llm_enabled: 是否啟用LLM
            llm_model: LLM模型名稱
            llm_base_url: LLM服務URL
            mixing_strategy: 混合策略 ("fixed_alpha", "adaptive", "llmpipe", "hybrid")
            alpha: 固定混合比例 (0-1)
            slope_threshold: 性能斜率閾值，低於此值觸發LLM
            llm_cooldown_episodes: LLM調用後的冷卻期
            min_episodes_before_llm: 使用LLM前的最小episode數

            # RL 參數
            replay_buffer_size: 經驗回放緩衝區大小
            gamma: 折扣因子
            tau: 軟更新係數
            batch_size: RL批次大小
            learning_rate: RL學習率
            actor_lr: Actor網絡學習率
            critic_lr: Critic網絡學習率
            hidden_dim: 網絡隱藏層維度

            # 其他參數
            seed: 隨機種子
            device: 計算設備
            verbose: 是否顯示詳細信息
            output_dir: 輸出目錄
        """

        # 存儲所有參數
        self.task_type = task_type
        self.max_trials = max_trials

        # LLM 參數
        self.llm_enabled = llm_enabled
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.mixing_strategy = mixing_strategy
        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.llm_cooldown_episodes = llm_cooldown_episodes
        self.min_episodes_before_llm = min_episodes_before_llm

        # RL 參數
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_dim = hidden_dim

        # 其他參數
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.output_dir = Path(output_dir)

        # 數據集參數
        self.dataset_info_csv_path = dataset_info_csv_path
        self.custom_prompt_template = custom_prompt_template
        self.dataset_name = dataset_name or f"FullControl_{task_type}"
        self.task_description = task_description

        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化組件
        self.environment = None
        self.hyperparameter_space = None
        self.optimizer = None
        self.results = None

        if verbose:
            print(f"🎛️  Full Control HPO initialized:")
            print(f"   Task: {task_type}")
            print(f"   Max trials: {max_trials}")
            print(f"   LLM: {'✅ Enabled' if llm_enabled else '❌ Disabled'}")
            if llm_enabled:
                print(f"   LLM Strategy: {mixing_strategy}")
                print(f"   Alpha: {alpha}")
                print(f"   Slope threshold: {slope_threshold}")
                print(f"   Cooldown episodes: {llm_cooldown_episodes}")
            print(f"   RL Buffer size: {replay_buffer_size}")
            print(f"   RL Learning rate: {learning_rate}")
            print(f"   Output: {output_dir}")

    def optimize(self,
                 X_train, y_train,
                 X_val=None, y_val=None,
                 X_test=None, y_test=None,
                 custom_space: Optional[HyperparameterSpace] = None,
                 custom_environment = None) -> Dict[str, Any]:
        """
        運行完全控制的優化

        Args:
            X_train, y_train: 訓練數據
            X_val, y_val: 驗證數據
            X_test, y_test: 測試數據
            custom_space: 自定義超參數空間
            custom_environment: 自定義環境

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
                print(f"📊 Auto-split data: train={len(X_train)}, val={len(X_val)}")

        # 設置環境
        if custom_environment:
            self.environment = custom_environment
        else:
            self.environment = TimeSeriesEnvironment(
                task_type=self.task_type,
                dataset_name=self.dataset_name
            )
            self.environment.set_data(X_train, y_train, X_val, y_val, X_test, y_test)
            self.environment.load_data()

        # 設置超參數空間
        if custom_space:
            self.hyperparameter_space = custom_space
        else:
            self.hyperparameter_space = self._create_default_space()

        # 創建完全配置的LLM配置
        llm_config = LLMEnhancedOptimizationConfig(
            # 基本RL參數 - 只傳遞 OptimizationConfig 支持的參數
            max_steps=self.max_trials,
            replay_buffer_size=self.replay_buffer_size,
            batch_size=self.batch_size,
            policy_learning_rate=self.actor_lr,
            value_learning_rate=self.critic_lr,
            
            # 設備和種子參數
            seed=self.seed,
            gpu_device=0,  # 使用默認GPU設備
            use_cuda=self.device != "cpu",
            verbose=self.verbose,

            # LLM參數
            enable_llm=self.llm_enabled,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            mixing_strategy=self.mixing_strategy,
            alpha=self.alpha,
            slope_threshold=self.slope_threshold,
            llm_cooldown_episodes=self.llm_cooldown_episodes,
            min_episodes_before_llm=self.min_episodes_before_llm,

            # 數據集和提示參數
            dataset_name=self.dataset_name,
            task_description=self.task_description or self.environment.get_llm_context(),
            custom_prompt_template=self.custom_prompt_template,
            dataset_info_csv_path=self.dataset_info_csv_path
        )

        # 創建優化器
        self.optimizer = LLMEnhancedMAT_HPO_Optimizer(
            environment=self.environment,
            hyperparameter_space=self.hyperparameter_space,
            config=llm_config,
            output_dir=str(self.output_dir)  # Pass custom output directory
        )

        if self.verbose:
            print(f"\n⚡ Starting full control optimization ({self.max_trials} trials)...")
            if self.llm_enabled:
                print(f"   🤖 LLM Strategy: {self.mixing_strategy}")
                print(f"   🎯 Alpha: {self.alpha}, Slope threshold: {self.slope_threshold}")
            print(f"   🧠 RL Buffer: {self.replay_buffer_size}, LR: {self.learning_rate}")

        # 運行優化
        import time
        start_time = time.time()
        self.results = self.optimizer.optimize()

        # 添加計時信息
        end_time = time.time()
        self.results['optimization_time'] = end_time - start_time
        self.results['trials_completed'] = self.max_trials

        if self.verbose:
            print(f"\n🎉 Full Control Optimization Complete!")
            print(f"   Time taken: {end_time - start_time:.1f} seconds")
            if 'best_performance' in self.results:
                perf = self.results['best_performance']
                print(f"\n🏆 Best Performance:")
                for metric, value in perf.items():
                    print(f"   {metric}: {value:.4f}")

            if 'best_hyperparameters' in self.results:
                print(f"\n⚙️ Best Hyperparameters:")
                for param, value in self.results['best_hyperparameters'].items():
                    print(f"   {param}: {value}")

        return self.results

    def _create_default_space(self) -> HyperparameterSpace:
        """創建默認超參數空間"""
        space = HyperparameterSpace()

        if "time_series" in self.task_type or "ecg" in self.task_type:
            # 時間序列特定參數
            space.add_discrete("hidden_size", [32, 64, 128, 256, 512], agent=0)
            space.add_continuous("learning_rate", 0.0001, 0.01, agent=1)
            space.add_discrete("batch_size", [16, 32, 64, 128], agent=1)
            space.add_continuous("dropout", 0.0, 0.5, agent=0)

            # 添加類別權重（假設9個類別，這是ICBEB數據集的標準）
            for i in range(9):  # ICBEB有9個類別
                space.add_continuous(f"class_weight_{i}", 0.5, 3.0, agent=2)
        else:
            # 通用分類參數
            space.add_continuous("learning_rate", 0.0001, 0.1, agent=0)
            space.add_discrete("batch_size", [16, 32, 64, 128, 256], agent=1)
            space.add_continuous("regularization", 1e-6, 1e-2, agent=2)

        return space

    def get_best_config(self) -> Dict[str, Any]:
        """獲取最佳超參數配置"""
        if self.results and 'best_hyperparameters' in self.results:
            return self.results['best_hyperparameters']
        return {}

    def get_optimization_stats(self) -> Dict[str, Any]:
        """獲取優化統計信息"""
        if not self.results:
            return {}

        stats = {
            'trials_completed': self.results.get('trials_completed', 0),
            'optimization_time': self.results.get('optimization_time', 0),
            'best_performance': self.results.get('best_performance', {}),
            'configuration': {
                'llm_enabled': self.llm_enabled,
                'mixing_strategy': self.mixing_strategy,
                'alpha': self.alpha,
                'slope_threshold': self.slope_threshold,
                'replay_buffer_size': self.replay_buffer_size,
                'learning_rate': self.learning_rate
            }
        }

        return stats


# 便利函數
def full_control_optimize(X_train, y_train, X_val=None, y_val=None,
                         task_type="time_series_classification",
                         max_trials=30,
                         llm_strategy="adaptive",
                         alpha=0.3,
                         slope_threshold=0.01,
                         **kwargs):
    """
    完全控制的一行式優化函數

    Args:
        X_train, y_train: 訓練數據
        X_val, y_val: 驗證數據
        task_type: 任務類型
        max_trials: 最大試驗次數
        llm_strategy: LLM策略
        alpha: 混合比例
        slope_threshold: 斜率閾值
        **kwargs: 其他所有參數

    Returns:
        最佳超參數配置
    """
    optimizer = FullControlHPO(
        task_type=task_type,
        max_trials=max_trials,
        mixing_strategy=llm_strategy,
        alpha=alpha,
        slope_threshold=slope_threshold,
        **kwargs
    )

    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    return optimizer.get_best_config()