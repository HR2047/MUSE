import sys
from pathlib import Path

# --- プロジェクトルートを動的に求めて sys.path に追加 ---
here = Path(__file__).resolve().parent        # scripts/ ディレクトリ
project_root = here.parent                    # project/ ディレクトリ
sys.path.insert(0, str(project_root))         # 検索パスの先頭に
# ----------------------------------------------------

from configs.config import GSASRecExperimentConfig
from ir_measures import nDCG, R

config = GSASRecExperimentConfig(
    dataset_name='book_interest_data_gsasrec_muse3',
    sequence_length=20,
    embedding_dim=64,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=3,
    gbce_t = 0.75,
    filter_rated=False, # 同じアイテムを推薦しないか
    recommendation_limit=4,
    metrics=[R@1, R@2], # 元コード
    val_metric = R@1,
    # 以下を追加
    num_user=482934,
    train_file_path="/home/hirosawa/research_m/MUSE_ver3/data/book_interest_data_gsasrec_muse3/train/input.txt",
    valid_input_path="/home/hirosawa/research_m/MUSE_ver3/data/book_interest_data_gsasrec_muse3/val/input.txt",
    valid_output_path="/home/hirosawa/research_m/MUSE_ver3/data/book_interest_data_gsasrec_muse3/val/output.txt",
)
