# Math Datasets

This folder contains math problem datasets for skill learning and evaluation.

## Available Datasets

- **aime24.json**: AIME 24 problems
- **aime25.json**: AIME 25 problems  
- **math500.json**: First 500 problems from MATH dataset
- **math1000.json**: First 1000 problems from MATH dataset
- **gsm8k.json**: GSM8K test set problems
- **gsm8k_train.json**: GSM8K training set problems

## Dataset Format

Each dataset is a JSON file with the following structure:

```json
[
  {
    "id": 1,
    "question": "Problem text here",
    "answer": "Final answer",
    "solution": "Step-by-step solution"
  },
  ...
]
```

## Download

### Automatic Download

To download datasets automatically:

```bash
# Install dependencies
pip install datasets requests

# Download default datasets (aime24, aime25, math500, gsm8k)
# The script will try Hugging Face first, then fall back to GitHub
python math_datasets/download_datasets.py

# Download specific datasets
python math_datasets/download_datasets.py --datasets aime24 aime25

# Download all available datasets
python math_datasets/download_datasets.py --all
```

### Manual Download

You can also manually download datasets from the kvpress repository:
https://github.com/minghui-liu/kvpress/tree/decode/reason

Place the JSON files in this `math_datasets/` directory.

## Usage in math_pipeline.py

Once datasets are downloaded, use them by name:

```bash
# Use default datasets (aime25 and math500)
python math_pipeline.py

# Use different datasets
python math_pipeline.py --dataset1 aime24 --dataset2 gsm8k
python math_pipeline.py --dataset1 gsm8k_train --dataset2 math1000
```

