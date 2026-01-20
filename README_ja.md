# MUSE: Multi-Stage User Interest Sequence Embedding

MUSEは、ユーザーの興味を動的なベクトルのシーケンスとしてモデル化するシーケンシャルレコメンデーションフレームワークです。レコメンデーションプロセスを、興味の抽出、興味の進化モデリング、そして最終的なアイテムレコメンデーションに分離しています。

## プロジェクト構成

コードベースは、処理パイプラインに対応する4つの主要なモジュールで構成されています。

### 1. `process_data/`
生のデータセットを後続のモデルで必要な形式に前処理するためのスクリプトが含まれています。
- データの分割、IDマッピング、フォーマット変換を処理します。

### 2. `interest_extraction/`
ユーザーのインタラクション履歴から複数の興味ベクトルを抽出するために、**ComiRec**モデルを実装しています。
- **主要ファイル**: `model.py` (ComiRecの実装)
- **出力**: 各ユーザーの興味ベクトルのシーケンスを生成します。

### 3. `predict_interest/`
ユーザーの興味の将来の軌跡を予測するために、**GSASRec**（Transformerベースのモデル）を実装しています。
- **`gsasrec_vector.py`**: 興味ベクトルのシーケンスを入力として受け取り、次の興味ベクトルを予測するコアモデル。
- **`gsasrec_label.py`**: アイテムIDで動作するバリアント（標準的なSASRecのアプローチ）。
- **`train_gsasrec_direct_interest.py`**: ベクトルベースモデルのトレーニングスクリプト。

### 4. `predict_item/`
予測された興味ベクトルに基づいて最終的なアイテムレコメンデーションを生成するための評価スクリプトが含まれています。
- **`evaluate_muse_direct_topk.py`**: 予測されたユーザーベクトルとアイテム埋め込みの類似度を計算し、Top-Kアイテムを推奨することでモデルを評価します。
- `ir_measures`を使用して、nDCGやRecallなどの指標を計算します。

## 必要要件

- Python 3.x
- **PyTorch**: GSASRec（興味予測）用。
- **TensorFlow 1.x**: ComiRec（興味抽出）および評価時のチェックポイント読み込み用。
- `ir_measures`: 評価指標用。
- `tqdm`, `numpy`, `pandas`

## 使用方法の概要

一般的なワークフローは以下の通りです。

1.  **データ処理**: `process_data/`内のスクリプトを実行して、データセットを準備します。
2.  **興味抽出**: `interest_extraction/`内のモデルをトレーニングして、ユーザーの興味埋め込みを取得します。
3.  **興味予測**: 抽出されたベクトルを使用して、`predict_interest/`内のGSASRecモデルをトレーニングします。
    ```bash
    python predict_interest/train_gsasrec_direct_interest.py --config config_file.py
    ```
4.  **評価**: `predict_item/`内の評価スクリプトを実行して、レコメンデーション性能を検証します。
    ```bash
    python predict_item/evaluate_muse_direct_topk.py --config config_file.py --dataset_stats_path path/to/stats.json
    ```

## 既知の問題と注意点

- **ハードコードされたパス**: いくつかのスクリプトには絶対パス（例: `/home/hirosawa/...`）が含まれています。**実行する前に、ローカル環境に合わせてこれらのパスを更新する必要があります**。
- **フレームワークの混在**: 評価スクリプトは、PyTorchとTensorFlowの両方のモデルを読み込むことがよくあります。環境が両方を処理できるように構成されていることを確認してください（例: 慎重なGPUメモリ管理）。
- **ファイルの欠落**: 一部のトレーニングスクリプトは、他のディレクトリ（例: `MUSE_ver4`）にある可能性のある`data_iterator`ファイルを参照しています。すべての依存関係が存在することを確認してください。
