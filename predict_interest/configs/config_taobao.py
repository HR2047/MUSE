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
    dataset_name='taobao_data_gsasrec_100',
    sequence_length=20,
    embedding_dim=64,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t = 0.75,
    val_metric=R@50,
    metrics=[nDCG@50, nDCG@20, nDCG@10, nDCG@5, nDCG@3, nDCG@1, R@50, R@20, R@10, R@5, R@3, R@1],
)
