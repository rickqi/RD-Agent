# Factor Discovery Loop - Performance Analysis & Optimization Plan

> Date: 2026-05-13
> Context: factor_run7 (6 loops, ~39 min), factor_run8 (5 loops, ~52 min), both crashed silently

---

## 1. Executive Summary

The RD-Agent factor discovery loop (Step 2 "running") is the dominant performance bottleneck, consuming **70-90% of total loop time**. Three compounding effects cause execution time to grow with each loop:

1. **SOTA factor re-processing** — regenerates ALL previously accepted factor data from scratch each loop
2. **Feature count growth** — backtest model trains on an ever-expanding feature set (Alpha158 + all accepted factors)
3. **IC deduplication scaling** — pairwise correlation calculation grows linearly with SOTA factor count

The root cause is `factor_proposal.py:117-119`, where `based_experiments` accumulates ALL successful experiments unboundedly. At loop N, there are N+1 experiments to process.

**Key finding**: A single qlib `qrun` data loading step takes **94 seconds** for 6,091 stocks, and LGBM training time scales linearly with feature count. With no feature capping, a 20-loop run can easily exceed 3 hours in backtest alone.

---

## 2. Per-Step Timing Data

### 2.1 factor_run8 Timing (Loops 6-10)

| Loop | Step 0 (direct_exp_gen) | Step 1 (coding) | Step 2 (running) | Step 3 (feedback) | Step 4 (record) | Total |
|------|------------------------|-----------------|------------------|-------------------|-----------------|-------|
| 6    | 7s                     | 136s            | ~583s            | 6s                | 1s              | ~11m  |
| 7    | 7s                     | 28s             | ~106s            | 3s                | 1s              | ~2.5m |
| 8    | 7s                     | 59s             | ~644s            | 6s                | 1s              | ~12m  |
| 9    | 7s                     | 15s             | ~222s            | 6s                | 1s              | ~4m   |
| 10   | 7s                     | 10s             | ~208s            | 6s                | 1s              | ~4m   |

### 2.2 Step 2 Sub-Timing Breakdown

| Loop | SOTA Factor Processing | New Factor Processing | Total Step 2 |
|------|----------------------|----------------------|--------------|
| 6    | 21s                  | 483s (incl. 94s qrun data loading) | ~504s |
| 7    | 20s                  | 66s                  | ~86s  |
| 8    | 96s                  | 548s                 | ~644s |
| 9    | 118s                 | 104s                 | ~222s |
| 10   | 24s                  | 184s                 | ~208s |
| 11   | 22s                  | **crashed**          | —     |

### 2.3 Key Observations

- **Step 2 is 70-90% of total loop time** in every loop
- **qrun data loading alone takes 94s** (loading 6,091 stocks from HDF5)
- **Step 1 (coding/CoSTEER) varies wildly**: 10-136s depending on iteration count
- **SOTA processing time spikes** at loops 8-9 (96s, 118s) when more factors are accepted
- **New factor processing is unpredictable**: 66-548s depending on code complexity

---

## 3. Root Cause Analysis

### 3.1 The Root Cause: Unbounded based_experiments Growth

**File**: `rdagent/scenarios/qlib/proposal/factor_proposal.py` lines 117-119

```python
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
```

This accumulates **ALL** successful (`feedback.decision=True`) `FactorExperiment` instances from the entire trace history. At loop N:
- `based_experiments` contains N+1 experiments (1 baseline + N accepted)
- Every subsequent step must process all N+1 experiments

### 3.2 Three Compounding Bottlenecks

#### Bottleneck 1: Factor Re-execution in process_factor_data()

**Path**: `factor_runner.py:93-95` → `utils.py:131-177`

```python
# factor_runner.py:90-95
sota_factor_experiments_list = [
    base_exp for base_exp in exp.based_experiments 
    if isinstance(base_exp, QlibFactorExperiment)
]
if len(sota_factor_experiments_list) > 1:
    SOTA_factor = process_factor_data(sota_factor_experiments_list)
```

Inside `process_factor_data()` (`utils.py:147-159`):
- Iterates through **every** experiment in the list
- For each: builds workspaces, creates execute calls, runs via multiprocessing
- Factor code execution IS cached via `@cache_with_pickle` on `FactorFBWorkspace.execute()` (`factor.py:106`)
- **However**: cache lookup, workspace construction, hashing, and `pd.concat` still cost O(N) overhead per loop

**Impact**: ~5-30s overhead at loop N, growing linearly. Mitigated by caching but not eliminated.

#### Bottleneck 2: deduplicate_new_factors() IC Calculation

**Path**: `factor_runner.py:35-44, 46-61, 106`

```python
# factor_runner.py:106
new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)

# factor_runner.py:35-44 — nested loop
def calculate_information_coefficient(self, concat_feature, sota_size, new_size):
    for col1 in range(sota_size):
        for col2 in range(sota_size, sota_size + new_size):
            res.loc[...] = concat_feature.iloc[:, col1].corr(
                concat_feature.iloc[:, col2])
```

- `sota_size` = total columns from ALL accepted experiments combined (grows with N)
- Each accepted experiment adds ~1-3 factor columns
- At loop N: sota_size ≈ 2N, new_size ≈ 2
- Per datetime group: O(2N × 2) = O(N) correlations
- Total: O(N × k × D) where D = number of trading dates (~3000)

**Impact**: Grows linearly with N. At loop 8 with ~16 SOTA columns, ~30-60s.

#### Bottleneck 3 (PRIMARY): Qlib Backtest with Growing Feature Set

**Path**: `factor_runner.py:111-172` → `workspace.py:47-51`

```python
# factor_runner.py:111 — combined_factors grows every loop
combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()

# factor_runner.py:127-130 — saved to parquet
combined_factors.to_parquet(target_path, engine="pyarrow")

# workspace.py:47-51 — qlib backtest executed
execute_qlib_log = qtde.check_output(
    local_path=str(self.workspace_path),
    entry=f"qrun {qlib_config_name}",
    env=run_env,
)
```

The qlib config (`conf_combined_factors.yaml`) uses `NestedDataLoader`:
```yaml
data_loader:
    class: NestedDataLoader
    kwargs:
        dataloader_l:
            - class: Alpha158DL          # 20 base features
            - class: StaticDataLoader    # loads combined_factors_df.parquet (GROWING)
```

**Total features = |base_features| + |combined_factors.columns|**
- Loop 0: ~20 features (Alpha20 only)
- Loop N: ~20 + 2*N features

**For LGBM**: Training time = O(n_features × n_samples × n_trees). With ~1M training samples, each additional feature adds measurable time.

**For Neural Network** (`conf_combined_factors_sota_model.yaml`):
```yaml
pt_model_kwargs: {
    "num_features": {{ num_features }}   # grows every loop
}
```
Larger input dimension = more model parameters = more compute per forward/backward pass.

**Impact**: **THE PRIMARY BOTTLENECK.** qrun data loading alone is 94s. Training time grows linearly (LGBM) or quadratically (NN) with feature count.

### 3.3 Time Complexity Summary

| Component | Complexity | Per-loop cost at loop N |
|-----------|-----------|------------------------|
| process_factor_data (SOTA re-exec) | O(N) experiments, cached | Linear (mitigated by cache) |
| process_factor_data (new factors) | O(k) new factors | Constant |
| deduplicate_new_factors | O(N × k × D) | Linear in N |
| combined_factors parquet save | O((20+2N) × S) | Linear in N |
| **qrun backtest (LGBM)** | **O((20+2N) × S × T)** | **Linear in N** |
| **qrun backtest (NN)** | **O((20+2N)² × S)** potentially | **Quadratic in N** |

Where: N = accepted loops, k = new factors per loop (~2), D = trading dates (~3000), S = total samples (~1M+), T = number of trees.

### 3.4 Why Loop 0 of run7 Took 692 Seconds

The baseline experiment runs Alpha20 features through qrun for CSI300 over 2008-2020 — already expensive. First run has no caching. The 28s at loop 6 run8 is fast because the baseline result was already cached from run7.

---

## 4. Caching Behavior (Current)

1. **`FactorFBWorkspace.execute()`** (`factor.py:106-107`): Cached by `@cache_with_pickle` using MD5 of factor code. Re-execution of identical code returns cached DataFrame. This is why process_factor_data doesn't blow up entirely.

2. **`QlibFactorRunner.develop()`** (`factor_runner.py:63`): Cached by `@cache_with_pickle(CachedRunner.get_cache_key, ...)`. The cache key hashes ALL tasks from ALL based_experiments. Each loop has unique new tasks, so this rarely helps.

3. **Factor code execution**: Cached per-factor. But workspace construction + pd.concat overhead is NOT cached.

4. **4.9GB pickle cache** accumulated on disk — potential memory/disk pressure contributor.

---

## 5. Optimization Plan

### 5.1 High-Impact Optimizations (Recommended First)

#### Opt-A: Cache Combined SOTA Factor Parquet (HIGHEST IMPACT)

**Current**: Every loop recomputes `process_factor_data(sota_factor_experiments_list)` from scratch by iterating through ALL accepted experiments.

**Proposed**: After computing SOTA_factor for the first time, save the combined parquet. On subsequent loops, load the cached parquet and only process the NEW experiment's factors, then concatenate.

**Implementation**:
- In `factor_runner.py:93-95`, add a cache check before calling `process_factor_data`
- Cache key = hash of all SOTA experiment task codes
- On cache hit: load previous SOTA parquet, only process new experiment
- On cache miss: full recomputation (first time only)

**Expected improvement**: Eliminates O(N) factor re-execution overhead. At loop 10, saves ~100s of SOTA processing time.

**Files to modify**: `rdagent/scenarios/qlib/developer/factor_runner.py`

#### Opt-B: Feature Selection Before Backtest (HIGHEST IMPACT)

**Current**: ALL accumulated features (Alpha20 + all accepted factors) are passed to qrun for LGBM/NN training.

**Proposed**: Before the backtest, apply a lightweight feature importance filter to keep only the top-K most predictive features (e.g., top 30). Use IC with label as the importance metric.

**Implementation**:
- In `factor_runner.py:120-127`, after `combined_factors` is built
- Compute IC of each feature column with the label (from qlib data)
- Keep only top-K features (K=30 or configurable)
- This caps the backtest cost regardless of loop count

**Expected improvement**: Caps backtest time to a constant regardless of loop count. At loop 10 with ~40 features, could reduce by 25-50%.

**Files to modify**: `rdagent/scenarios/qlib/developer/factor_runner.py`

#### Opt-C: Cap based_experiments (SIMPLE FIX)

**Current**: `factor_proposal.py:117-119` accumulates ALL successful experiments.

**Proposed**: Limit to only the last K successful experiments (e.g., K=5 or K=3).

**Implementation**:
```python
# Before:
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]

# After:
all_sota = [t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)]
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + all_sota[-MAX_SOTA_EXPERIMENTS:]
```

**Expected improvement**: Bounds all O(N) operations to O(K). Simple change, immediate benefit.

**Files to modify**: `rdagent/scenarios/qlib/proposal/factor_proposal.py`

### 5.2 Medium-Impact Optimizations

#### Opt-D: Incremental IC Deduplication

**Current**: `deduplicate_new_factors()` computes IC of new factors against ALL SOTA columns.

**Proposed**: Only compute IC against the latest batch of accepted factors. Previously accepted factors were already deduplicated against their predecessors.

**Implementation**:
- In `factor_runner.py:106`, pass only the last batch of SOTA factors instead of all
- This reduces O(N × k) to O(k × k_prev) per loop

**Expected improvement**: At loop 10, reduces from ~20×2 to ~2×2 IC calculations.

**Files to modify**: `rdagent/scenarios/qlib/developer/factor_runner.py`

#### Opt-E: Warm-Start Qlib Data Loading

**Current**: Every qrun invocation loads ALL stock data from HDF5 (94s for 6,091 stocks).

**Proposed**: Pre-load qlib data once and keep in memory/shared memory across loops.

**Implementation**:
- Use qlib's `D.cache()` mechanism to pre-cache dataset
- Or use a persistent qlib init at the start of the loop and reuse across loops

**Expected improvement**: Eliminates the 94s data loading per qrun invocation. However, this requires significant refactoring of the subprocess-based execution model.

**Files to modify**: `rdagent/scenarios/qlib/experiment/workspace.py`, possibly qlib config

### 5.3 Advanced Optimizations (Future Work)

#### Opt-F: Incremental Model Training

- For LGBM: Use previous model's trees as warm-start
- For NN: Transfer learning from previous model with expanded input layer
- Requires deep integration with qlib's training pipeline

#### Opt-G: Parallel Factor Processing

- Process multiple new factors in parallel using asyncio/multiprocessing
- Current code already uses `multiprocessing_wrapper` but only for factor code execution

#### Opt-H: Memory Management

- Periodic pickle cache cleanup (4.9GB accumulated)
- WSL2 `.wslconfig` memory limit configuration
- Crash-recovery wrapper script for long-running processes

---

## 6. Recommended Implementation Order

| Priority | Optimization | Effort | Impact | Risk |
|----------|-------------|--------|--------|------|
| 1 | Opt-C: Cap based_experiments | Low (5 lines) | High | Low |
| 2 | Opt-A: Cache SOTA parquet | Medium (30 lines) | High | Low |
| 3 | Opt-B: Feature selection | Medium (40 lines) | Highest | Medium |
| 4 | Opt-D: Incremental IC dedup | Low (10 lines) | Medium | Low |
| 5 | Opt-E: Warm-start data loading | High (refactor) | High | High |
| 6 | Opt-H: Memory management | Low (config) | Medium | Low |

### Phase 1 (Quick Wins): Opt-C + Opt-D + Opt-H
- ~20 lines of code changes
- Bounds all O(N) scaling to O(K)
- Can be implemented and tested in 30 minutes

### Phase 2 (Core Optimization): Opt-A + Opt-B
- ~70 lines of code changes
- Eliminates the primary bottleneck
- Requires careful testing to ensure factor quality isn't degraded by feature selection

### Phase 3 (Advanced): Opt-E + Opt-F
- Significant refactoring required
- Only needed for very long runs (50+ loops)

---

## 7. Silent Crash Analysis

Both run7 (loop 6) and run8 (loop 11) crashed silently — no OOM error, no traceback, process just disappeared.

**Suspected cause**: WSL2 memory reclamation. The process peaked at ~3.0GB RAM. WSL2 may reclaim memory aggressively when system pressure is high, killing processes without warning.

**Recommended mitigations**:
1. Add `.wslconfig` with explicit memory limit (e.g., 12GB of 16GB)
2. Wrap the process with a crash-recovery script that auto-resumes from checkpoint
3. Periodically clean pickle cache to reduce memory pressure
4. Monitor memory usage within the loop and add explicit GC between loops

**Crash-recovery script concept**:
```bash
#!/bin/bash
while true; do
    rdagent fin_factor --step-begin-from $LOOP --loop-n $REMAINING
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed successfully"
        break
    fi
    echo "Crashed with exit code $EXIT_CODE. Resuming..."
    # Parse latest checkpoint and update LOOP/REMAINING
    LOOP=$(python parse_checkpoint.py)
    REMAINING=$((20 - LOOP))
done
```

---

## 8. Key Files Reference

| File | Role | Lines of Interest |
|------|------|-------------------|
| `rdagent/scenarios/qlib/proposal/factor_proposal.py` | based_experiments accumulation | 117-119 |
| `rdagent/scenarios/qlib/developer/factor_runner.py` | SOTA processing, dedup, backtest | 35-44, 46-61, 63, 90-172 |
| `rdagent/scenarios/qlib/developer/utils.py` | process_factor_data() | 131-177 |
| `rdagent/scenarios/qlib/experiment/workspace.py` | qrun execution | 47-51 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml` | qlib LGBM config | 12-27 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml` | qlib NN config | 81 |
| `rdagent/core/utils.py` | cache_with_pickle, multiprocessing_wrapper | 124, 156 |
| `rdagent/components/runner/__init__.py` | CachedRunner.get_cache_key | 8-14 |
