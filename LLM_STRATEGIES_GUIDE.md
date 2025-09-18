# LLM策略正確使用指南

基於論文 [arXiv:2507.13712](https://arxiv.org/abs/2507.13712) 的LLM增強超參數優化策略指南。

## 支援的LLM策略

### 1. `fixed_alpha` - 固定混合比例

**描述：** 使用固定的混合比例結合LLM和RL的建議。

**使用場景：**
- 新手用戶，希望簡單穩定的結果
- 已知最佳混合比例的情況
- 希望可預測行為的生產環境

**關鍵參數：**
- `alpha`: 混合比例 (0.0-1.0)
  - 0.3 表示 30% LLM + 70% RL
  - 推薦值：0.2-0.4

**示例：**
```python
from MAT_HPO_LIB import LLMEnhancedHPO

optimizer = LLMEnhancedHPO(
    task_type="time_series_classification",
    max_trials=30,
    llm_enabled=True,
    llm_strategy="fixed_alpha",
    llm_config={
        'alpha': 0.3,  # 30% LLM建議 + 70% RL建議
        'model': 'llama3.2:3b'
    }
)
```

### 2. `adaptive` - 自適應觸發 (論文核心方法)

**描述：** 監控RL性能改善斜率，當斜率低於閾值時觸發LLM介入。

**理論基礎：** 基於論文arXiv:2507.13712的adaptive trigger機制，通過監控強化學習的性能改善趨勢來決定何時使用LLM。

**使用場景：**
- 希望智能自動調整的情況
- 不確定最佳介入時機
- 研究和實驗場景
- 追求最優性能的生產環境

**關鍵參數：**
- `performance_threshold`: RL性能斜率閾值
  - 當RL性能改善斜率 < 閾值時，觸發LLM
  - 推薦值：0.005-0.02
  - 較小值 = 更敏感，更頻繁使用LLM
  - 較大值 = 更保守，較少使用LLM

**示例：**
```python
from MAT_HPO_LIB import LLMEnhancedHPO

optimizer = LLMEnhancedHPO(
    task_type="time_series_classification",
    max_trials=30,
    llm_enabled=True,
    llm_strategy="adaptive",  # 自適應策略
    llm_config={
        'performance_threshold': 0.01,  # RL性能斜率閾值
        'min_episodes': 5,              # 開始監控前的最少episodes
        'cooldown': 3,                  # LLM介入後的冷卻期
        'model': 'llama3.2:3b'
    }
)
```

## 參數詳細說明

### 通用LLM參數

- **`llm_model`**: LLM模型名稱
  - 預設：`"llama3.2:3b"`
  - 支援Ollama所有模型

- **`llm_cooldown`**: LLM冷卻期
  - 預設：5 episodes
  - LLM介入後等待的episodes數

### Adaptive策略特定參數

- **`performance_threshold`**: 性能斜率閾值
  - 預設：0.01
  - 範圍：0.001-0.1
  - 影響：決定LLM介入的敏感度

- **`min_episodes_before_llm`**: 最少episodes
  - 預設：5
  - 開始監控RL性能前的最少訓練episodes

## 策略選擇建議

### 選擇 `fixed_alpha` 當：
- ✅ 需要可預測的行為
- ✅ 有明確的混合比例偏好
- ✅ 計算資源有限
- ✅ 生產環境要求穩定性

### 選擇 `adaptive` 當：
- ✅ 追求最優性能
- ✅ 希望系統自動調整
- ✅ 進行研究或實驗
- ✅ 有充足計算資源
- ✅ 不確定最佳介入策略

## 完整使用範例

### SPNV2 ECG分類範例

```bash
# 使用固定混合比例
python "2. SPL_HPO_Complete.py" --dataset ICBEB --fold 1 --steps 20 --gpu 0 \
  --llm_enabled --llm_strategy fixed_alpha --llm_alpha 0.3

# 使用自適應策略 (推薦)
python "2. SPL_HPO_Complete.py" --dataset ICBEB --fold 1 --steps 20 --gpu 0 \
  --llm_enabled --llm_strategy adaptive
```

### 程式碼範例

```python
# 基本自適應策略
optimizer = LLMEnhancedHPO(
    task_type="ecg_classification",
    max_trials=25,
    llm_enabled=True,
    llm_strategy="adaptive"  # 使用預設的自適應參數
)

# 進階自適應策略
llm_config = {
    'performance_threshold': 0.008,  # 較敏感的閾值
    'min_episodes': 3,               # 較早開始監控
    'cooldown': 2,                   # 較短冷卻期
    'model': 'llama3.2:3b',
    'buffer_size': 1000,             # RL參數
    'learning_rate': 0.001
}

optimizer = LLMEnhancedHPO(
    task_type="time_series_classification",
    max_trials=30,
    llm_enabled=True,
    llm_strategy="adaptive",
    llm_config=llm_config
)
```

## 性能調優建議

### Adaptive策略調優：

1. **performance_threshold調整：**
   - 開始值：0.01
   - 如果LLM使用太少：降低至0.005-0.008
   - 如果LLM使用太多：提高至0.015-0.02

2. **min_episodes_before_llm調整：**
   - 簡單問題：3-5 episodes
   - 複雜問題：5-8 episodes

3. **llm_cooldown調整：**
   - 快速實驗：2-3 episodes
   - 穩定性優先：5-7 episodes

### Fixed Alpha策略調優：

1. **alpha值選擇：**
   - 保守策略：0.2 (20% LLM)
   - 平衡策略：0.3 (30% LLM)
   - 積極策略：0.4 (40% LLM)

## 常見問題

**Q: 何時使用哪種策略？**
A: 新手或生產環境使用fixed_alpha；研究或追求最優性能使用adaptive。

**Q: adaptive策略不工作怎麼辦？**
A: 檢查performance_threshold是否太高，嘗試降低至0.005。

**Q: 如何知道LLM是否在工作？**
A: 開啟verbose=True，查看"LLM decision"和"RL decision"的日誌。

**Q: 可以同時使用兩種策略嗎？**
A: 不可以，每次優化只能選擇一種策略。

---

**重要提醒：** 此實現嚴格遵循論文arXiv:2507.13712的定義，確保學術和實用的一致性。