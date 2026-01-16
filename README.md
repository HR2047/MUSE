# MUSE: Multi-Stage User Interest Sequence Embedding

MUSE is a sequential recommendation framework that models user interests as a dynamic sequence of vectors. It decouples the recommendation process into interest extraction, interest evolution modeling, and final item recommendation.

## Project Structure

The codebase is organized into four main modules corresponding to the processing pipeline:

### 1. `process_data/`
Contains scripts for preprocessing raw datasets into the formats required by the subsequent models.
- Handles data splitting, ID mapping, and format conversion.

### 2. `interest_extraction/`
Implements the **ComiRec** model to extract multiple interest vectors from user interaction history.
- **Key File**: `model.py` (ComiRec implementation)
- **Output**: Generates sequence of interest vectors for each user.

### 3. `predict_interest/`
Implements **GSASRec** (Transformer-based model) to predict the future trajectory of user interests.
- **`gsasrec_vector.py`**: The core model that takes a sequence of interest vectors as input and predicts the next interest vector.
- **`gsasrec_label.py`**: A variant that operates on item IDs (standard SASRec approach).
- **`train_gsasrec_direct_interest.py`**: Training script for the vector-based model.

### 4. `predict_item/`
Contains evaluation scripts to generate final item recommendations based on the predicted interest vectors.
- **`evaluate_muse_direct_topk.py`**: Evaluates the model by calculating the similarity between the predicted user vector and item embeddings to recommend top-K items.
- Uses `ir_measures` for calculating metrics like nDCG and Recall.

## Requirements

- Python 3.x
- **PyTorch**: For GSASRec (Interest Prediction).
- **TensorFlow 1.x**: For ComiRec (Interest Extraction) and loading its checkpoints during evaluation.
- `ir_measures`: For evaluation metrics.
- `tqdm`, `numpy`, `pandas`

## Usage Overview

The general workflow is as follows:

1.  **Data Processing**: Run scripts in `process_data/` to prepare your dataset.
2.  **Interest Extraction**: Train the model in `interest_extraction/` to obtain user interest embeddings.
3.  **Interest Prediction**: Train the GSASRec model in `predict_interest/` using the extracted vectors.
    ```bash
    python predict_interest/train_gsasrec_direct_interest.py --config config_file.py
    ```
4.  **Evaluation**: Run evaluation scripts in `predict_item/` to verify recommendation performance.
    ```bash
    python predict_item/evaluate_muse_direct_topk.py --config config_file.py --dataset_stats_path path/to/stats.json
    ```

## Known Issues & Notes

- **Hardcoded Paths**: Several scripts contain absolute paths (e.g., `/home/hirosawa/...`). **You must update these paths** to match your local environment before running.
- **Mixed Frameworks**: The evaluation scripts often load both PyTorch and TensorFlow models. Ensure your environment is configured to handle both (e.g., careful GPU memory management).
- **Missing Files**: Some training scripts reference `data_iterator` files that may be located in other directories (e.g., `MUSE_ver4`). Ensure all dependencies are present.
