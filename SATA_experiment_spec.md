# SATA Experimental Codebase — Specification for Implementation

## Overview

This document specifies the full experimental codebase for an honours research project titled **"Faithful Out-of-Distribution Generalisation via Demonstration Design"**. The project evaluates whether in-context demonstration design can improve both OOD robustness and decision faithfulness for frozen LLMs on tabular classification tasks. The core contribution is **SATA (Shift-Aware Task Adapter)**, a small auxiliary transformer that learns to select and reweight in-context demonstrations in a query-conditioned manner.

The codebase is structured as a series of Jupyter notebooks, each handling one phase of the pipeline, supported by shared utility modules. The project has **two experimental arms** that run in parallel:

- **Real arm**: frozen LLM inference on TableShift benchmarks (RQ1, RQ3-internal-consistency). No training. Starts immediately.
- **Synthetic arm**: task generation, SATA training, and evaluation with known ground truth (RQ2, RQ4, RQ3-correctness). Requires the generator to be built first.

**10-week timeline**. Code quality matters for the thesis appendix but engineering elegance is secondary to getting reproducible results.

---

## Project Structure

```
sata-project/
├── notebooks/
│   ├── 01_tableshift_setup.ipynb          # Real arm: data loading, preprocessing, pilot
│   ├── 02_real_arm_baselines.ipynb         # Real arm: all Family A conditions on real data
│   ├── 03_faithfulness_real.ipynb          # Real arm: π_self, π_behav, Spearman ρ
│   ├── 04_task_generator.ipynb            # Synthetic arm: build + validate generator
│   ├── 05_sata_train.ipynb                # Synthetic arm: SATA architecture + training
│   ├── 06_synthetic_evaluation.ipynb      # Synthetic arm: RQ2 grid, RQ4, ablations
│   ├── 07_uncertainty_rauc.ipynb          # Pass 2: R-AUC and F1@95% (both arms)
│   └── 08_figures_and_tables.ipynb        # Final: all thesis figures, tables, stats
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py                   # Synthetic task generator
│   │   ├── tableshift_loader.py           # TableShift data loading + preprocessing
│   │   └── serialisation.py               # Row → text serialisation for LLM prompts
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sata.py                        # SATA transformer architecture
│   │   ├── sata_targets.py                # Target score computation from ground truth
│   │   └── sata_train.py                  # Training loop (KL loss)
│   │
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── random_select.py               # Random-k baseline
│   │   ├── similarity_select.py           # Cosine similarity retrieval
│   │   ├── label_diversity.py             # Protocol A: balanced class sampling
│   │   ├── feature_range.py               # Protocol B: quantile-bin coverage
│   │   ├── rule_diversity.py              # Protocol C: decision-tree leaf sampling
│   │   ├── counter_spurious.py            # Protocol D: spurious-correlation breakers
│   │   └── sata_select.py                 # SATA-based selection (wraps sata.py)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── llm_runner.py                  # vLLM interface: prompt → prediction + logprobs
│   │   └── prompts.py                     # System instruction and prompt templates
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── accuracy.py                    # Accuracy, macro-F1, shift gap
│   │   ├── faithfulness.py                # π_self, π_behav (hot-deck LOO), Spearman ρ
│   │   ├── faithfulness_correctness.py    # π_true vs π_behav (synthetic only)
│   │   ├── uncertainty.py                 # Normalised logprob confidence, R-AUC, F1@95%
│   │   └── bootstrap.py                   # Bootstrap CIs for any metric
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                      # All hyperparameters, paths, seeds
│       └── results_schema.py             # Parquet schema definition
│
├── configs/
│   └── default.yaml                       # Master config (models, k, seeds, paths)
│
├── results/                               # All output parquets, one row per prediction
├── figures/                               # Generated thesis figures
└── requirements.txt
```

---

## Week-0 Decisions (Hardcoded in `configs/default.yaml`)

```yaml
# Models
base_llms:
  - name: "Llama-3.1-8B-Instruct"
    path: "meta-llama/Llama-3.1-8B-Instruct"  # or local path
  - name: "Qwen2.5-7B-Instruct"               # second model for robustness
    path: "Qwen/Qwen2.5-7B-Instruct"

# Serving
vllm:
  tensor_parallel: 1
  gpu_memory_utilisation: 0.90
  max_model_len: 4096

# Demo selection
k_primary: 8
k_sensitivity: 16       # one dataset only
pool_size: 256           # real arm demo pool
seed_accuracy: [42, 123, 456, 789, 1024]    # 5 seeds for accuracy runs
seed_faithfulness: [42, 123, 456]            # 3 seeds for faithfulness

# Evaluation
test_rows_id: 500
test_rows_ood: 500
faithfulness_subset: 200

# SATA
sata:
  d_model: 128
  n_heads: 4
  n_layers: 4
  max_demos: 64
  max_features: 16      # pad/truncate to this
  lr: 1e-4
  epochs: 50
  batch_size: 64

# Generator
generator:
  n_features: 10
  n_causal_range: [3, 5]
  n_train_tasks: 2000
  n_val_tasks: 200
  n_test_tasks: 200
  n_heldout_family_tasks: 50
  rule_families: ["linear", "threshold", "tree", "sparse_interaction"]
  heldout_family: "sparse_interaction"
  spurious_strength_range: [0.70, 0.95]
  label_noise: 0.05
  demos_per_task: 64
  queries_per_env: 32
  environments: ["id", "covariate", "spurious_reversal", "extrapolation", "missing_feature", "mechanism"]

# Results schema
results_columns:
  - arm           # "real" or "synthetic"
  - dataset       # e.g. "acs_income" or "task_0042"
  - environment   # e.g. "id", "ood", "spurious_reversal"
  - model         # e.g. "llama-3.1-8b"
  - method        # e.g. "random", "sata+counter_spurious"
  - seed          # demo sampling seed
  - query_id      # unique per query
  - prediction    # model's predicted label
  - label         # ground truth
  - logprob_0     # normalised logprob of label token 0
  - logprob_1     # normalised logprob of label token 1
  - demo_ids      # list of selected demo indices
  - k             # number of demos used
```

---

## Notebook 01: TableShift Setup (`01_tableshift_setup.ipynb`)

**Purpose**: Load TableShift, select 3 datasets, preprocess, verify OOD splits exist and show meaningful shift gaps.

### Steps

1. **Install TableShift**: `pip install tableshift` (or clone from `github.com/mlfoundations/tableshift`). Check which datasets are publicly downloadable without credentialed access (avoid MIMIC-derived).

2. **Select 3 datasets**. Candidates ranked by published shift gap and public accessibility:
   - ACS Income (geographic shift — predict income ≥$50k, shift across US states)
   - ACS Public Coverage (demographic shift)
   - BRFSS Diabetes (temporal/geographic shift)
   - ANES Voting (temporal shift)
   
   Selection criteria: public access, binary classification, ≤15 usable features after reduction, nontrivial published shift gap.

3. **Preprocessing per dataset**:
   - Load train / ID-test / OOD-test splits using TableShift API.
   - Feature reduction: select top 10–15 features by mutual information with the label on the training split. Document which features were kept and why.
   - Handle missing values: mode imputation for categorical, median for continuous. Record imputation strategy.
   - Create **demo pool**: randomly sample 256 rows from training split (stratified by label). Fix this pool across all conditions and seeds — the pool is constant, only the selection from it varies.
   - Create **test sets**: 500 rows from ID-test, 500 from OOD-test (or all available if <500).
   - Save everything as parquet files with clear naming.

4. **Serialisation template** (`src/data/serialisation.py`):
   ```python
   def serialise_row(features: dict, label: str = None) -> str:
       """Convert a tabular row to a text string for the LLM prompt.
       
       Format: "FeatureName1: value1; FeatureName2: value2; ... → Label"
       Feature order is fixed per dataset (alphabetical) and recorded in config.
       If label is None (query row), end with "→"
       """
   ```
   Feature order is a nuisance variable — freeze it alphabetically per dataset and never randomise it.

5. **Pilot run**: zero-shot and random-8 on one dataset with one model. Verify the pipeline works end-to-end: serialisation → prompt → vLLM → prediction → logprobs → accuracy. This is a smoke test, not a result.

### Output
- `data/real/{dataset_name}/train_pool.parquet` (256 rows)
- `data/real/{dataset_name}/test_id.parquet` (500 rows)
- `data/real/{dataset_name}/test_ood.parquet` (500 rows)
- `data/real/{dataset_name}/feature_list.json` (ordered feature names)
- `data/real/{dataset_name}/label_tokens.json` (e.g. `["Approved", "Denied"]`)

---

## Notebook 02: Real Arm Baselines (`02_real_arm_baselines.ipynb`)

**Purpose**: Run all Family A demo-selection conditions on all 3 real datasets, both ID and OOD test sets, 2 models, 5 seeds.

### Conditions (7 total)

1. **Zero-shot**: no demonstrations, just system instruction + query.
2. **Random-k**: sample k=8 demos uniformly from the pool.
3. **Similarity-k**: embed all serialised rows (demo pool + query) with a small sentence encoder (`all-MiniLM-L6-v2`), select k=8 demos with highest cosine similarity to the query.
4. **Label diversity**: stratified sampling — k/2 demos from each class.
5. **Feature-range diversity**: for each of the top 3 continuous features (by MI), bin into 3 quantiles. Sample demos to cover all bins. Fill remaining slots randomly.
6. **Rule diversity**: fit a depth-3 decision tree on the training split. Identify leaf membership for each demo in the pool. Sample demos to cover all populated leaves. Fill remaining slots randomly.
7. **Counter-spurious diversity**: identify the 2-3 features most correlated with both the label and the shift variable (the variable that defines the TableShift domain split). Over-sample demos from "minority cells" — rows where the proxy-label correlation is broken. E.g., if high-income correlates with label=1 and the shift is geographic, find demos where high-income → label=0.

### Implementation notes

- **Prompt template** (`src/inference/prompts.py`):
  ```
  System: You are a classifier. Given the features of an individual, predict {task_description}.
  Respond with exactly one word: {label_0} or {label_1}.

  {serialised_demo_1}
  {serialised_demo_2}
  ...
  {serialised_demo_k}

  {serialised_query} →
  ```

- **LLM runner** (`src/inference/llm_runner.py`):
  - Use vLLM's `SamplingParams` with `logprobs=True`, `max_tokens=1`, `temperature=0`.
  - After generation, extract logprobs for the two label tokens. Apply constrained softmax:
    ```python
    def get_confidence(logprobs_dict, label_tokens):
        lp0 = logprobs_dict.get(label_tokens[0], -100)
        lp1 = logprobs_dict.get(label_tokens[1], -100)
        p0 = np.exp(lp0) / (np.exp(lp0) + np.exp(lp1))
        p1 = 1 - p0
        pred = label_tokens[0] if p0 >= p1 else label_tokens[1]
        confidence = max(p0, p1)
        return pred, confidence, p0, p1
    ```
  - If the model's top token is not in the label set (rare with constrained decoding, but handle it), log it as `INVALID` and exclude from accuracy but include in the count.

- **Batch processing**: for each (dataset, model, condition, seed), generate all 1000 prompts (500 ID + 500 OOD), batch them through vLLM, collect results.

- **Save every prediction** as one row in the results parquet:
  ```
  arm=real, dataset=acs_income, environment=ood, model=llama-3.1-8b, 
  method=counter_spurious, seed=42, query_id=347, 
  prediction=Approved, label=Denied, logprob_0=-0.41, logprob_1=-1.12,
  demo_ids=[12,45,67,89,102,156,201,238], k=8
  ```

### Compute budget estimate
- 7 conditions × 5 seeds × 1000 rows × 3 datasets × 2 models = 210,000 LLM calls.
- At ~100 tokens per prompt (short for k=8), batched vLLM on 8B model: ~2-4 hours per dataset per model on a single GPU. Total: ~12-24 hours wall time.

### Output
- `results/real_arm_baselines.parquet`
- Summary table: accuracy, macro-F1, shift gap per (dataset, model, condition), averaged over seeds with std.

---

## Notebook 03: Faithfulness on Real Data (`03_faithfulness_real.ipynb`)

**Purpose**: Compute internal-consistency faithfulness ρ(π_self, π_behav) on real datasets. This answers RQ3's internal-consistency component.

### Step 1: Elicit π_self (self-reported feature ranking)

For each (dataset, condition, seed), prompt the LLM:
```
System: You are analyzing a {task_description} prediction task.
Given the following features: {feature_list}
Rank these features from most important to least important for predicting {label_description}.
Respond with a comma-separated list of feature names, most important first.
```

Parse the output into an ordered list. This is π_self. Run this once per (dataset, condition, seed) — it doesn't depend on individual queries.

### Step 2: Compute π_behav (behavioural feature ranking via leave-one-out)

For each feature j in the feature set:
1. Take the faithfulness test subset (200 rows from OOD-test, fixed across conditions).
2. Replace feature j's value in every row using **kNN hot-deck imputation**: for each row, find the 5 nearest neighbours in the training pool (by Euclidean distance on all features *except* j), and randomly sample one neighbour's value for feature j. This preserves feature correlations better than marginal replacement.
3. Run the full inference pipeline (with the selected demos) on the modified test set.
4. Compute the accuracy drop: Δ_j = accuracy(original) − accuracy(modified).

Rank features by Δ_j descending. This is π_behav. The feature whose removal hurts most is ranked first.

### Step 3: Compute ρ and bootstrap CIs

```python
from scipy.stats import spearmanr

rho, pval = spearmanr(pi_self_ranks, pi_behav_ranks)

# Bootstrap
n_bootstrap = 1000
rho_samples = []
for _ in range(n_bootstrap):
    # Resample the 200 test rows with replacement
    idx = np.random.choice(200, size=200, replace=True)
    # Recompute pi_behav on resampled rows
    delta_j_boot = [compute_accuracy_drop(modified_preds[j][idx], labels[idx]) for j in features]
    pi_behav_boot = np.argsort(-np.array(delta_j_boot))
    rho_boot, _ = spearmanr(pi_self_ranks, rank_from_order(pi_behav_boot))
    rho_samples.append(rho_boot)

ci_low = np.percentile(rho_samples, 2.5)
ci_high = np.percentile(rho_samples, 97.5)
```

### Conditions to evaluate (3 only, to keep compute manageable)
- Random-k
- Best-performing protocol from Notebook 02
- (Later, Week 8) SATA if real-data transfer works

### Compute budget
- Per condition: 200 rows × ~12 features × 1 forward pass = 2,400 calls for behavioural ranking.
- 3 conditions × 3 seeds × 3 datasets × 2 models = ~130,000 calls.
- Still comfortable for vLLM batching.

### Output
- `results/faithfulness_real.parquet` (one row per feature per condition per dataset per seed)
- Summary: ρ ± CI per (dataset, model, condition)

---

## Notebook 04: Synthetic Task Generator (`04_task_generator.ipynb`)

**Purpose**: Build, validate, and freeze the synthetic task generator. This is SATA's training data source and the testbed for RQ2/RQ4.

### Generator specification (`src/data/generator.py`)

```python
class SyntheticTask:
    """One synthetic tabular classification task with known ground truth."""
    
    def __init__(self, task_id, rule_family, causal_features, coefficients,
                 spurious_strength, n_features=10, label_noise=0.05):
        self.task_id = task_id
        self.rule_family = rule_family          # "linear" | "threshold" | "tree" | "sparse_interaction"
        self.causal_features = causal_features  # list of indices, e.g. [0, 2, 4]
        self.coefficients = coefficients        # rule parameters
        self.spurious_idx = 8                   # always feature 8
        self.noise_idx = 9                      # always feature 9
        self.spurious_strength = spurious_strength
        self.label_noise = label_noise
        self.n_features = n_features
    
    def generate_environment(self, env_type, n_samples):
        """Generate (X, y, metadata) for a given environment type."""
        # Returns:
        #   X: (n_samples, n_features) array
        #   y: (n_samples,) binary labels
        #   metadata: dict with per-sample info (regime, is_counter_spurious, etc.)
```

### Rule families

Each family defines how y is computed from the causal features:

**Linear**: `logit = sum(coeff_i * X[:, causal_i])`, `y = (sigmoid(logit) > 0.5)`

**Threshold**: `y = (X[:, causal_0] > threshold_0) AND/OR (X[:, causal_1] > threshold_1)`

**Tree (depth 2-3)**: nested if-else on 2-3 causal features with different thresholds at each branch. Defines 4-8 "regimes" — each leaf of the tree is a regime.

**Sparse interaction**: `y` depends on a product or ratio of two causal features exceeding a threshold, e.g. `y = (X[:, 0] * X[:, 3] > threshold)`. Rest of causal features are decoys with small coefficients.

### Environment generation

For each task, generate 6 environments:

| Environment | What changes | How |
|---|---|---|
| `id` | Nothing | Same distribution as training |
| `covariate` | Feature marginals | Shift means of 2-3 features by 1-2 std |
| `spurious_reversal` | Spurious correlation | Flip spurious_strength from e.g. 0.9 to 0.1 |
| `extrapolation` | Feature range | Sample features from 2-3× the training range |
| `missing_feature` | One causal feature | Set one causal feature to 0 (or dataset mean) |
| `mechanism` | Rule coefficients | Perturb coefficients by ±50% |

### Per-task data structure

For each task × environment:
- Demo pool: 64 rows with metadata tags:
  - `regime`: which leaf/region of the decision rule this demo falls in
  - `is_counter_spurious`: True if the demo's spurious feature value disagrees with the label (i.e., breaks the shortcut)
  - `spurious_consistent`: True if spurious feature agrees with label
- Query set: 32 rows with the same metadata

### Task sampling

```python
def generate_task_suite(config):
    tasks = []
    for i in range(config.n_train_tasks):
        family = random.choice(config.rule_families)  # exclude heldout_family
        n_causal = random.randint(*config.n_causal_range)
        causal_feats = sorted(random.sample(range(8), n_causal))
        coeffs = np.random.randn(n_causal) * 2  # scale for meaningful separation
        spur_strength = random.uniform(*config.spurious_strength_range)
        
        task = SyntheticTask(
            task_id=f"train_{i:04d}", rule_family=family,
            causal_features=causal_feats, coefficients=coeffs,
            spurious_strength=spur_strength
        )
        tasks.append(task)
    
    # Held-out family tasks (e.g., sparse_interaction only)
    for i in range(config.n_heldout_family_tasks):
        # Same as above but family = config.heldout_family
        ...
    
    return tasks
```

### XGBoost validation gate (Week 2-3)

For each of ~50 randomly sampled tasks from the training set:
1. Fit `XGBClassifier(max_depth=4, n_estimators=100)` on the `id` environment's 64 demos.
2. Evaluate on 32 queries from each of the 6 environments.
3. Check:
   - ID accuracy > 80% (task is learnable)
   - Spurious-reversal accuracy < 60% (spurious feature is strong enough to mislead)
   - Mechanism-shift accuracy < ID accuracy by ≥15 points (mechanism shift is real)
4. If the pattern doesn't hold for a majority of sampled tasks, adjust generator parameters (spurious_strength_range, coefficient scale, label_noise) and re-run.

**Gate criterion**: ≥80% of sampled tasks show the expected degradation profile. Only then freeze the generator.

### Output
- `data/synthetic/tasks_train/` — 2000 task directories, each containing environment parquets
- `data/synthetic/tasks_val/` — 200 tasks
- `data/synthetic/tasks_test/` — 200 tasks  
- `data/synthetic/tasks_heldout_family/` — 50 tasks
- `data/synthetic/generator_config.json` — frozen generator parameters
- `data/synthetic/xgboost_validation.parquet` — gate results

---

## Notebook 05: SATA Architecture and Training (`05_sata_train.ipynb`)

**Purpose**: Define, train, and validate SATA on synthetic tasks.

### Architecture (`src/models/sata.py`)

```python
class SATA(nn.Module):
    """Shift-Aware Task Adapter.
    
    Input: 
        demo_features: (batch, n_demos, n_features)  — candidate demonstrations
        demo_labels:   (batch, n_demos)               — demo labels (0 or 1)
        query_features:(batch, n_features)             — the query row
    
    Output:
        scores: (batch, n_demos)  — relevance score per demo, sums to 1 via softmax
    """
    
    def __init__(self, n_features, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        # Per-feature linear embedding
        self.feature_embed = nn.Linear(n_features, d_model)
        # Label embedding (2 classes)
        self.label_embed = nn.Embedding(2, d_model)
        # Learnable query token type embedding
        self.query_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.demo_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Scoring head: one scalar per demo position
        self.score_head = nn.Linear(d_model, 1)
    
    def forward(self, demo_features, demo_labels, query_features):
        batch_size, n_demos, n_feat = demo_features.shape
        
        # Embed demos: feature embedding + label embedding
        demo_emb = self.feature_embed(demo_features) + self.label_embed(demo_labels)
        demo_emb = demo_emb + self.demo_type_embed
        
        # Embed query: feature embedding only (no label)
        query_emb = self.feature_embed(query_features).unsqueeze(1)
        query_emb = query_emb + self.query_type_embed
        
        # Concatenate: [demo_1, demo_2, ..., demo_n, query]
        sequence = torch.cat([demo_emb, query_emb], dim=1)  # (batch, n_demos+1, d_model)
        
        # Self-attention (all tokens attend to all)
        encoded = self.transformer(sequence)
        
        # Extract demo positions only (not query)
        demo_encoded = encoded[:, :n_demos, :]  # (batch, n_demos, d_model)
        
        # Score each demo
        logits = self.score_head(demo_encoded).squeeze(-1)  # (batch, n_demos)
        scores = F.softmax(logits, dim=-1)
        
        return scores
```

### Target score computation (`src/models/sata_targets.py`)

For each query in a synthetic task, compute the target relevance score for each demo:

```python
def compute_target_scores(task, query_metadata, demo_metadata, temperature=1.0):
    """
    Assign target relevance weights based on generator ground truth.
    
    High weight:
        - Demos in the same decision regime as the query
        - Counter-spurious demos (break the shortcut)
    Low weight:
        - Demos only predictive via spurious feature
        - Demos from irrelevant regimes
    
    Returns: (n_demos,) array, normalised to sum to 1
    """
    scores = np.zeros(len(demo_metadata))
    
    for i, demo in enumerate(demo_metadata):
        score = 0.0
        
        # Same regime bonus
        if demo['regime'] == query_metadata['regime']:
            score += 2.0
        
        # Counter-spurious bonus
        if demo['is_counter_spurious']:
            score += 1.5
        
        # Correct label bonus (mild)
        # Not too strong — we want structural alignment, not just label matching
        if demo['label'] == query_metadata['label']:
            score += 0.5
        
        # Penalty for spurious-only demos
        if demo['spurious_consistent'] and demo['regime'] != query_metadata['regime']:
            score += 0.1  # near-zero but not exactly zero for numerical stability
        
        scores[i] = score
    
    # Normalise to distribution via softmax with temperature
    scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
    return scores
```

### Training loop (`src/models/sata_train.py`)

```python
def train_sata(model, train_tasks, val_tasks, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for task in train_tasks:
            for env_type in config.environments:
                env_data = task.generate_environment(env_type, n_samples=96)
                demos = env_data[:64]   # demo pool
                queries = env_data[64:] # query set
                
                for query in queries:
                    # Compute target scores from ground truth
                    target = compute_target_scores(task, query.metadata, demos.metadata)
                    target_tensor = torch.tensor(target, dtype=torch.float32)
                    
                    # Forward pass
                    pred_scores = model(demos.features, demos.labels, query.features)
                    
                    # KL divergence loss
                    loss = F.kl_div(
                        pred_scores.log(),    # SATA's predicted log-distribution
                        target_tensor,         # target distribution
                        reduction='batchmean'
                    )
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
        
        # Validation: check if SATA selects demos that lead to better
        # XGBoost accuracy than random selection (proxy, no LLM needed)
        val_score = evaluate_sata_proxy(model, val_tasks)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, val_proxy={val_score:.4f}")
```

**Note on batching**: the pseudocode above is per-query for clarity. In practice, batch across queries within a task for efficiency. Each task provides 6 envs × 32 queries = 192 training examples.

### SATA ablation: query-agnostic variant

```python
class SATAQueryAgnostic(SATA):
    """Ablation: mask the query token so scores don't depend on query identity."""
    
    def forward(self, demo_features, demo_labels, query_features):
        # Replace query with a zero vector — scores depend only on demo pool
        dummy_query = torch.zeros_like(query_features)
        return super().forward(demo_features, demo_labels, dummy_query)
```

This ablation isolates whether per-query conditioning matters — the core SATA vs ICR differentiator.

### Gate 2 (end of Week 5)

Compare on validation tasks:
- SATA top-k selection vs. best protocol selection vs. random selection
- Metric: XGBoost accuracy when trained on the selected k demos and tested on queries (proxy — doesn't require LLM)

If SATA doesn't beat the best protocol on validation: RQ4 becomes a rigorous negative result. Document why and pivot to ablation analysis.

### Output
- `models/sata_best.pt` — best checkpoint by validation loss
- `models/sata_query_agnostic.pt` — ablation checkpoint
- `results/sata_training_log.parquet` — loss curves, validation metrics per epoch

---

## Notebook 06: Synthetic Arm Evaluation (`06_synthetic_evaluation.ipynb`)

**Purpose**: Run the full evaluation matrix on synthetic tasks. Answers RQ2 and RQ4.

### Conditions

All Family A methods, now on synthetic tasks with the frozen LLM:
1. Zero-shot
2. Random-k
3. Similarity-k
4. Label diversity
5. Feature-range diversity
6. Rule diversity (using ground-truth regime labels from the generator — no need for a decision tree on synthetic data)
7. Counter-spurious diversity (using ground-truth is_counter_spurious tags)
8. Best protocol + SATA selection
9. SATA alone (selects from full pool without protocol pre-filtering)
10. SATA query-agnostic ablation

### RQ2 evaluation (protocol × shift type grid)

For each condition × shift type, compute accuracy on the 200 held-out test tasks:

```
               | id  | cov  | spur_rev | extrap | missing | mech |
random         | ... | ...  | ...      | ...    | ...     | ...  |
label_div      | ... | ...  | ...      | ...    | ...     | ...  |
feat_range     | ... | ...  | ...      | ...    | ...     | ...  |
rule_div       | ... | ...  | ...      | ...    | ...     | ...  |
counter_spur   | ... | ...  | ...      | ...    | ...     | ...  |
```

**RQ2 success**: interaction effect — different protocols win on different shift types (e.g. feature-range best on covariate, counter-spurious best on spurious-reversal). Shift type is known by construction — no need for DISDE decomposition.

### RQ4 evaluation (SATA vs protocols)

Compare SATA + best protocol vs. best protocol alone on:
- Accuracy (all shift types, held-out test tasks + held-out family tasks)
- Correctness-of-reliance faithfulness: ρ(π_behav, π_true) where π_true comes from the generator's known causal features

### RQ3 correctness-of-reliance (synthetic only)

On synthetic tasks, you can compute whether the model relies on the **correct** features:

```python
# π_true: rank features by their true importance (from generator)
# causal features get high rank, spurious gets low rank, noise gets lowest
pi_true = rank_by_true_importance(task.causal_features, task.coefficients)

# π_behav: computed via LOO ablation as in Notebook 03
pi_behav = compute_behavioural_ranking(model_predictions, ...)

rho_correctness = spearmanr(pi_true, pi_behav)
```

### Output
- `results/synthetic_evaluation.parquet`
- `results/rq2_grid.parquet` — protocol × shift type accuracy grid
- `results/rq4_comparison.parquet` — SATA vs protocol comparison
- `results/faithfulness_synthetic.parquet` — correctness-of-reliance ρ values

---

## Notebook 07: Uncertainty and R-AUC (`07_uncertainty_rauc.ipynb`)

**Purpose**: Pass-2 metrics. Compute R-AUC and F1@95% using normalised logprob confidence. Runs on results already collected in Notebooks 02 and 06.

### R-AUC computation

```python
def compute_rauc(predictions, labels, confidences):
    """
    Error-retention curve: sort by ascending confidence,
    progressively replace least-confident predictions with ground truth,
    compute error at each retention level.
    """
    n = len(predictions)
    sorted_idx = np.argsort(confidences)  # least confident first
    
    errors = (predictions != labels).astype(float)
    
    retention_levels = np.linspace(0, 1, 101)  # 0% to 100% retention
    error_at_retention = []
    
    for r in retention_levels:
        n_retain = int(r * n)
        if n_retain == 0:
            error_at_retention.append(0.0)
            continue
        # Keep the n_retain most confident predictions
        retained_idx = sorted_idx[n - n_retain:]
        error_rate = errors[retained_idx].mean()
        error_at_retention.append(error_rate)
    
    rauc = np.trapz(error_at_retention, retention_levels)
    return rauc, retention_levels, error_at_retention
```

### F1@95% computation

```python
def compute_f1_at_retention(predictions, labels, confidences, retention=0.95):
    """F1 score using only the top 95% most confident predictions."""
    n_retain = int(retention * len(predictions))
    sorted_idx = np.argsort(confidences)
    retained_idx = sorted_idx[len(predictions) - n_retain:]
    
    from sklearn.metrics import f1_score
    return f1_score(labels[retained_idx], predictions[retained_idx], average='macro')
```

### Confidence values

The logprob_0 and logprob_1 columns from the results parquets already contain the normalised logprobs. Confidence = `max(exp(logprob_0), exp(logprob_1))` for each prediction.

### Output
- `results/uncertainty_metrics.parquet` — R-AUC and F1@95% per condition
- Retention curve plots for key comparisons

---

## Notebook 08: Figures and Tables (`08_figures_and_tables.ipynb`)

**Purpose**: Generate all thesis figures and summary tables from the results parquets.

### Figures to generate

1. **RQ1 bar chart**: ID vs OOD accuracy across 3 datasets, zero-shot and random-k. Shows the problem exists.
2. **RQ2 heatmap**: protocol × shift type accuracy grid (synthetic arm). Shows interaction effect.
3. **RQ3 scatter**: Δaccuracy vs Δρ across all conditions. Shows whether accuracy and faithfulness dissociate.
4. **RQ4 grouped bars**: SATA + protocol vs protocol alone vs random, on accuracy and ρ, with error bars from bootstrap CIs.
5. **Retention curves**: R-AUC plots for best vs worst conditions.
6. **SATA ablation table**: full SATA vs query-agnostic vs varying k vs varying pool size.
7. **Training curves**: SATA loss and validation metric over epochs.

### Statistical tests

- Paired Wilcoxon signed-rank test across tasks for protocol comparisons.
- Holm correction for multiple comparisons when comparing all protocols.
- All CIs from bootstrap (already computed in evaluation notebooks).

### Output
- `figures/rq1_ood_degradation.pdf`
- `figures/rq2_protocol_shift_heatmap.pdf`
- `figures/rq3_accuracy_vs_faithfulness.pdf`
- `figures/rq4_sata_comparison.pdf`
- `figures/retention_curves.pdf`
- `figures/sata_ablations.pdf`
- `tables/` — LaTeX table source files for thesis

---

## Key Implementation Notes

### Data contamination guarantee
SATA is meta-trained exclusively on synthetic tasks from the generator. It is never exposed to any TableShift data during training. At evaluation time on TableShift, SATA receives standardised feature vectors (zero mean, unit variance within each task's demo pool). This design ensures performance gains reflect genuine transfer of the learned selection strategy.

### Reproducibility
- All random seeds are set via `config.yaml` and passed through every function.
- The demo pool is sampled once per (dataset, seed) and saved; selection methods operate on the saved pool.
- The generator is frozen after the XGBoost validation gate; all subsequent notebooks use the frozen tasks.
- Results are append-only parquets; re-running a notebook appends rather than overwrites unless explicitly cleared.

### Dependencies
```
torch>=2.0
transformers
vllm
datasets
tableshift
scikit-learn
xgboost
numpy
pandas
scipy
matplotlib
seaborn
sentence-transformers  # for similarity-k embedding
pyarrow                # for parquet I/O
pyyaml
```

---

## Week-by-Week Execution Order

| Week | Notebooks | Gate |
|---|---|---|
| 1 | 01 (TableShift setup + pilot) | Pipeline works end-to-end? |
| 2 | 02 (real arm baselines, 1 dataset), 04 (generator build) | — |
| 3 | 02 (remaining datasets), 04 (XGBoost validation) | **Gate 1**: vanilla ICL degrades OOD on real data? Generator passes validation? |
| 4 | 03 (faithfulness, start), 05 (SATA architecture + training start) | — |
| 5 | 05 (SATA training complete) | **Gate 2**: SATA > best protocol on synthetic validation? |
| 6 | 06 (synthetic evaluation: RQ2 grid, RQ4, ablations) | — |
| 7 | 06 (remaining ablations, correctness-of-reliance), 07 (R-AUC pass) | — |
| 8 | 07 (complete), SATA real-data transfer (stretch) | — |
| 9 | 08 (all figures, tables, stats) | — |
| 10 | Buffer, thesis writing | — |
