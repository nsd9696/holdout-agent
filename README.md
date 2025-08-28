# Holdout Agent

Holdout Agent is a tool for intelligently splitting machine learning benchmark datasets into held-in and held-out portions. It can use OpenAI Agents to intelligently partition data based on goals, or execute the splitting logic directly.

## Key Features

- **Intelligent Data Splitting**: Uses OpenAI Agents to split data according to specific goals
- **Multiple Benchmark Support**: HellaSWAG, ARC Challenge, MMLU, KMMLU, and more
- **Difficulty-Based Splitting**: Configure difficulty distribution across 1-5 levels
- **Topic-Based Filtering**: Select data related to specific topics
- **Elasticsearch Integration**: Efficient data retrieval through vector search
- **Advanced Search Strategies**: Multi-stage, hybrid, and adaptive search for better candidate retrieval
- **Score Threshold Filtering**: Configurable similarity thresholds for quality control

## Project Structure

```
holdout-agent/
├── main.py                 # Main execution file
├── holdout_agent.py        # CLI interface and core logic
├── agent_runner.py         # OpenAI Agent runner
├── split_engine.py         # Data splitting engine
├── meta_infer.py           # Metadata inference (topics, difficulty)
├── es_index.py             # Elasticsearch index management
├── ingest.py               # Data ingestion and indexing
├── config.py               # Configuration and environment variables
├── run_example.py          # Usage examples
└── test_split.py           # Splitting tests
```

## Installation & Setup

### 1. Install Dependencies

```bash
# Install using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
export ES_ENDPOINT="http://localhost:9200"
export ES_USER="elastic"
export ES_PASS="your_password"
export ES_INDEX="llm_bench_rows"
export OPENAI_API_KEY="your_openai_api_key"

# Advanced search configuration (optional)
export SEARCH_MIN_SCORE_THRESHOLD="0.1"
export SEARCH_FALLBACK_MULTIPLIER="1.5"
export SEARCH_MAX_ATTEMPTS="5"
```

### 3. Elasticsearch Setup

Elasticsearch must be running. Using Docker:

```bash
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.18.1
```

## Usage

### Basic Usage

```bash
python holdout_agent.py --goal "Evaluate commonsense reasoning" --total 300 --heldin-ratio 0.5
```

### Advanced Options

```bash
python holdout_agent.py \
  --goal "Evaluate commonsense reasoning and science questions" \
  --benchmarks hellaswag arc_challenge kmmlu \
  --topics commonsense science \
  --total 500 \
  --heldin-ratio 0.6 \
  --difficulty-mix "1:0.1,2:0.2,3:0.4,4:0.2,5:0.1" \
  --mode agent
```

### Parameter Description

- `--goal`: Evaluation goal (required)
- `--benchmarks`: List of benchmarks to use (optional)
- `--topics`: Topics to filter by (optional)
- `--total`: Total number of data points
- `--heldin-ratio`: Held-in ratio (0.0 ~ 1.0)
- `--difficulty-mix`: Difficulty distribution (e.g., "1:0.05,2:0.15,3:0.5,4:0.2,5:0.1")
- `--mode`: Execution mode (`agent` or `direct`)

## Code Examples

### Using in Python Directly

```python
from holdout_agent import main
from agent_runner import run_agent_sync

# Run in Agent mode
payload = {
    "goal": "Evaluate commonsense reasoning",
    "total": 300,
    "difficulty_mix": {"1": 0.05, "2": 0.15, "3": 0.5, "4": 0.2, "5": 0.1},
    "topics": ["commonsense"],
    "benchmarks": ["hellaswag"],
    "heldin_ratio": 0.5
}

result = run_agent_sync(payload)
print(result)
```

### Data Ingestion and Indexing

```python
from es_index import get_es_client, ensure_index
from ingest import ingest_all

# Create Elasticsearch index
es = get_es_client()
ensure_index(es)

# Ingest data
ingest_all(es)
```

## Output Files

- `held_in.jsonl`: Held-in dataset
- `held_out.jsonl`: Held-out dataset

Each file is saved in JSONL format, where each line represents one data point.

## Supported Benchmarks

| Benchmark     | HuggingFace Repository | Split      | Subset |
| ------------- | ---------------------- | ---------- | ------ |
| HellaSWAG     | hellaswag              | validation | -      |
| ARC Challenge | ai2_arc                | validation | -      |
| MMLU          | cais/mmlu              | validation | -      |
| KMMLU         | HAERAE-HUB/KMMLU       | train      | -      |

## Evaluation

The project includes comprehensive evaluation frameworks to assess the quality of the holdout agent's data splits:

### 1. Distributional Parity, Topic Coverage, and Difficulty Alignment Evaluation

This evaluation provides a comprehensive analysis of held-in and held-out splits using OpenAI embeddings and statistical methods.

**Metrics:**

- **Distributional Parity**: FID-style Gaussian distance, KS tests on top-K PCA components (Holm-Bonferroni)
- **Topic Coverage**: Jensen-Shannon divergence, Shannon entropy, topic diversity analysis
- **Difficulty Alignment**: Distribution comparison, mean difficulty analysis, difficulty balance assessment

**Usage:**

```bash
# Run with default settings
python evaluation_distributional_parity_topic_difficulty.py

# Run with custom files and model
python evaluation_distributional_parity_topic_difficulty.py \
  --held_in held_in_custom.jsonl \
  --held_out held_out_custom.jsonl \
  --openai_model text-embedding-3-large \
  --outdir ./custom_eval_results

# Use different OpenAI embedding model
python evaluation_distributional_parity_topic_difficulty.py --openai_model text-embedding-3-small

# Customize PCA components and batch size
python evaluation_distributional_parity_topic_difficulty.py --topk_pcs 20 --batch_size 64
```

**Command Line Options:**

- `--held_in`: Path to held-in data file (default: `held_in.jsonl`)
- `--held_out`: Path to held-out data file (default: `held_out.jsonl`)
- `--openai_model`: OpenAI embedding model to use (default: `text-embedding-3-large`)
- `--dimensions`: Optional dimensionality reduction if supported by model (default: None)
- `--batch_size`: Batch size for OpenAI API calls (default: 128)
- `--topk_pcs`: Number of top PCA components for KS tests (default: 10)
- `--outdir`: Output directory for results (default: `./eval_simple_out`)

**Expected Output:**

- Three comprehensive tables: `table_parity.md/csv`, `table_topic.md/csv`, `table_difficulty.md/csv`
- **Parity**: FID-style distance, KS test results on PCA components
- **Topics**: JS divergence, entropy, topic diversity counts
- **Difficulty**: Distribution mix (1-5 bins), mean difficulty, "Out_harder?" assessment
- Results saved in both CSV and Markdown formats

### Running Evaluations

```bash
# Run the comprehensive evaluation (recommended)
python evaluation_distributional_parity_topic_difficulty.py

# Run with custom parameters
python evaluation_distributional_parity_topic_difficulty.py \
  --input my_held_in.jsonl \
  --output my_held_out.jsonl \
  --embed-model sentence-transformers/all-mpnet-base-v2 \
  --output-dir ./my_results
```

### Evaluation Results Interpretation

**Distributional Parity:**

- **Good splits**: Low MMD (~0.001-0.01), low Sliced Wasserstein distance (~0.1-0.5)
- **Poor splits**: High MMD (>0.1), high Wasserstein distance (>2.0)
- **P-values**: Higher p-values suggest more similar distributions

**Topic Coverage:**

- **JS Divergence**: Lower values indicate more similar topic distributions
- **Coverage**: Higher values show better topic representation
- **Entropy**: Balanced entropy suggests good topic diversity

**Difficulty Alignment:**

- **EMD**: Lower values indicate more similar difficulty distributions
- **Mean Difficulty**: Compare held-in vs held-out average difficulty
- **Distribution Mix**: Check if difficulty proportions are maintained

## Development

### Running Tests

```bash
pytest test_split.py
```

## License

This project is distributed under the MIT License.

## Contributing

Bug reports, feature requests, and pull requests are welcome. Please create an issue before contributing.
