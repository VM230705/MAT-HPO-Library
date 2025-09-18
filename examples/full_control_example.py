#!/usr/bin/env python3
"""
Full Control HPO Example - 完全參數控制示例

這個示例展示如何使用FullControlHPO進行完全參數化的超參數優化，
包含所有LLM策略和RL參數的細粒度控制。
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add MAT_HPO_LIB to path
sys.path.insert(0, '/home/vm230705/MAT-HPO-Library')

def example_fixed_alpha():
    """示例1: 固定混合比例策略"""
    print("=" * 70)
    print("示例1: Fixed Alpha Strategy - 固定混合比例")
    print("=" * 70)

    from MAT_HPO_LIB import FullControlHPO

    # 創建數據
    X, y = make_classification(n_samples=400, n_features=15, n_classes=3,
                             n_informative=12, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"數據集: {X_train.shape[0]} 訓練樣本, {X_val.shape[0]} 驗證樣本, {len(np.unique(y))} 個類別")

    # 完全控制的HPO - 固定混合比例
    optimizer = FullControlHPO(
        task_type="time_series_classification",
        max_trials=8,

        # LLM 參數完全控制
        llm_enabled=True,
        mixing_strategy="fixed_alpha",      # 固定混合策略
        alpha=0.4,                         # 40% LLM + 60% RL
        llm_model="llama3.2:3b",
        llm_cooldown_episodes=3,           # LLM冷卻期3個episode

        # RL 參數完全控制
        replay_buffer_size=500,            # 經驗回放緩衝區
        learning_rate=0.002,               # RL學習率
        actor_lr=0.001,                    # Actor學習率
        critic_lr=0.003,                   # Critic學習率
        gamma=0.95,                        # 折扣因子
        tau=0.005,                         # 軟更新係數
        hidden_dim=128,                    # 隱藏層維度

        # 其他參數
        seed=42,
        verbose=True,
        output_dir="./full_control_fixed_alpha_results"
    )

    print("\\n🎯 使用固定混合比例 (40% LLM + 60% RL)")
    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    print(f"\\n📊 最佳性能:")
    for metric, value in results['best_performance'].items():
        print(f"   {metric}: {value:.4f}")

    return results

def example_adaptive_strategy():
    """示例2: 自適應策略 - 監控性能斜率"""
    print("\\n" + "=" * 70)
    print("示例2: Adaptive Strategy - 性能斜率監控")
    print("=" * 70)

    from MAT_HPO_LIB import FullControlHPO

    # 創建更複雜的數據
    X, y = make_classification(n_samples=600, n_features=20, n_classes=4,
                             n_informative=16, random_state=123)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

    print(f"數據集: {X_train.shape[0]} 訓練樣本, {X_val.shape[0]} 驗證樣本")

    # 自適應策略 - 監控RL性能斜率
    optimizer = FullControlHPO(
        task_type="ecg_classification",
        max_trials=10,

        # LLM 自適應參數
        llm_enabled=True,
        mixing_strategy="adaptive",         # 自適應策略
        slope_threshold=0.005,              # 斜率閾值 - 更敏感
        min_episodes_before_llm=3,          # 最少3個episode才檢查斜率
        llm_cooldown_episodes=4,            # 冷卻期4個episode

        # RL 參數調優
        replay_buffer_size=800,
        learning_rate=0.0015,
        batch_size=32,                      # 較小的批次
        gamma=0.98,
        tau=0.002,

        # 其他參數
        seed=123,
        verbose=True,
        output_dir="./full_control_adaptive_results"
    )

    print(f"\\n🎯 使用自適應策略 (斜率閾值: {optimizer.slope_threshold})")
    print(f"   當RL性能斜率 < {optimizer.slope_threshold} 時，增加LLM使用")
    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    stats = optimizer.get_optimization_stats()
    print(f"\\n📈 優化統計:")
    print(f"   總耗時: {stats['optimization_time']:.2f} 秒")
    print(f"   試驗完成: {stats['trials_completed']}")

    return results

def example_adaptive_slope_monitoring():
    """示例3: Adaptive策略 - 基於論文arXiv:2507.13712的RL性能斜率監控"""
    print("\\n" + "=" * 70)
    print("示例3: Adaptive Strategy - 基於論文的RL性能斜率監控")
    print("=" * 70)

    from MAT_HPO_LIB import FullControlHPO

    # 創建時間序列數據
    np.random.seed(456)
    n_samples, seq_length = 300, 80
    X = np.random.randn(n_samples, seq_length, 1)

    # 添加模式
    for i in range(n_samples):
        if i % 3 == 0:  # 類別0: 正弦波
            X[i, :, 0] += np.sin(np.linspace(0, 4*np.pi, seq_length))
        elif i % 3 == 1:  # 類別1: 餘弦波
            X[i, :, 0] += np.cos(np.linspace(0, 4*np.pi, seq_length))
        else:  # 類別2: 鋸齒波
            X[i, :, 0] += np.linspace(-1, 1, seq_length)

    y = np.array([i % 3 for i in range(n_samples)])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=456)

    print(f"時間序列數據: {X_train.shape}")

    # LLaPipe策略 - 基於論文的實現
    optimizer = FullControlHPO(
        task_type="time_series_classification",
        max_trials=12,

        # Adaptive 特定參數
        llm_enabled=True,
        mixing_strategy="adaptive",         # 自適應策略
        slope_threshold=0.01,               # RL性能斜率閾值 (論文建議)
        min_episodes_before_llm=5,          # 開始監控前的最少episode數

        # 針對時間序列的RL參數
        replay_buffer_size=1000,
        learning_rate=0.001,
        actor_lr=0.0008,
        critic_lr=0.002,
        hidden_dim=256,                     # 較大的隱藏層

        # 設備和輸出
        device="auto",
        seed=456,
        verbose=True,
        output_dir="./full_control_llmpipe_results"
    )

    print(f"\\n🎯 使用Adaptive策略 (基於論文arXiv:2507.13712)")
    print(f"   監控RL學習曲線斜率，當 < {optimizer.slope_threshold} 時觸發LLM")
    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    # 獲取最佳配置
    best_config = optimizer.get_best_config()
    print(f"\\n⚙️ 最佳超參數配置:")
    for param, value in best_config.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.6f}")
        else:
            print(f"   {param}: {value}")

    return results

def example_custom_hyperparameter_space():
    """示例4: 自定義超參數空間"""
    print("\\n" + "=" * 70)
    print("示例4: Custom Hyperparameter Space - 自定義搜索空間")
    print("=" * 70)

    from MAT_HPO_LIB import FullControlHPO, HyperparameterSpace

    # 創建數據
    X, y = make_classification(n_samples=400, n_features=12, n_classes=2,
                             n_informative=10, random_state=789)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=789)

    # 定義自定義超參數空間
    custom_space = HyperparameterSpace()

    # Agent 0: 模型架構參數
    custom_space.add_discrete('hidden_size', [64, 128, 256, 512], agent=0)
    custom_space.add_continuous('dropout', 0.1, 0.7, agent=0)
    custom_space.add_discrete('num_layers', [2, 3, 4], agent=0)

    # Agent 1: 訓練參數
    custom_space.add_continuous('learning_rate', 0.0001, 0.05, agent=1)
    custom_space.add_discrete('batch_size', [8, 16, 32, 64, 128], agent=1)
    custom_space.add_continuous('weight_decay', 1e-6, 1e-2, agent=1)

    # Agent 2: 優化參數
    custom_space.add_continuous('momentum', 0.8, 0.99, agent=2)
    custom_space.add_discrete('optimizer_type', ['adam', 'sgd', 'rmsprop'], agent=2)

    print(f"🔧 自定義超參數空間:")
    print(f"   Agent 0 (架構): hidden_size, dropout, num_layers")
    print(f"   Agent 1 (訓練): learning_rate, batch_size, weight_decay")
    print(f"   Agent 2 (優化): momentum, optimizer_type")

    # 使用混合策略
    optimizer = FullControlHPO(
        task_type="classification",
        max_trials=10,

        # 混合策略參數
        llm_enabled=True,
        mixing_strategy="hybrid",           # 混合策略
        alpha=0.25,                        # 基礎混合比例
        slope_threshold=0.008,

        # 精細調整的RL參數
        replay_buffer_size=600,
        learning_rate=0.0012,
        gamma=0.97,
        tau=0.003,

        seed=789,
        verbose=True,
        output_dir="./full_control_custom_space_results"
    )

    print(f"\\n🎯 使用混合策略 + 自定義超參數空間")
    results = optimizer.optimize(X_train, y_train, X_val, y_val,
                               custom_space=custom_space)

    # 展示最佳配置
    best_config = optimizer.get_best_config()
    print(f"\\n🏆 最佳配置 (自定義空間):")
    for param, value in best_config.items():
        print(f"   {param}: {value}")

    return results

def example_convenience_function():
    """示例5: 便利函數使用"""
    print("\\n" + "=" * 70)
    print("示例5: Convenience Function - 便利函數")
    print("=" * 70)

    from MAT_HPO_LIB.full_control_hpo import full_control_optimize

    # 創建簡單數據
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2,
                             n_informative=6, random_state=999)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=999)

    print(f"數據集: {X_train.shape[0]} 訓練樣本")

    # 使用便利函數
    print(f"\\n🚀 使用便利函數進行一行式優化")
    best_config = full_control_optimize(
        X_train, y_train, X_val, y_val,
        task_type="classification",
        max_trials=6,
        llm_strategy="adaptive",
        alpha=0.35,
        slope_threshold=0.012,
        learning_rate=0.0018,
        replay_buffer_size=400,
        verbose=True
    )

    print(f"\\n🎯 最佳配置 (便利函數):")
    for param, value in best_config.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.6f}")
        else:
            print(f"   {param}: {value}")

    return best_config

def main():
    """運行所有示例"""
    print("Full Control HPO 完整示例")
    print("=" * 70)
    print("這些示例展示如何使用FullControlHPO進行完全參數化的超參數優化")
    print("包含所有LLM策略和RL參數的細粒度控制")
    print()

    try:
        # 運行所有示例
        print("🎯 開始運行完全控制HPO示例...")

        result1 = example_fixed_alpha()
        result2 = example_adaptive_strategy()
        result3 = example_adaptive_slope_monitoring()
        result4 = example_custom_hyperparameter_space()
        result5 = example_convenience_function()

        print("\\n" + "🎉" * 25)
        print("所有Full Control HPO示例成功完成!")
        print("🎉" * 25)

        # 總結對比
        print("\\n📊 策略效果對比:")
        print(f"Fixed Alpha:     F1={result1['best_performance'].get('f1', 0):.4f}")
        print(f"Adaptive Basic:  F1={result2['best_performance'].get('f1', 0):.4f}")
        print(f"Adaptive Slope:  F1={result3['best_performance'].get('f1', 0):.4f}")
        print(f"Custom Space:    F1={result4['best_performance'].get('f1', 0):.4f}")

    except KeyboardInterrupt:
        print("\\n❌ 示例被用戶中斷")
    except Exception as e:
        print(f"\\n❌ 運行示例時出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()