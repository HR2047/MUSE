from ir_measures import nDCG, R

class GSASRecExperimentConfig(object):
    def __init__(self, dataset_name, sequence_length=20, embedding_dim=64, train_batch_size=128,
                             num_heads=4, num_blocks=3, 
                             dropout_rate=0.0,
                             negs_per_pos=256,
                             max_epochs=10000,
                             max_batches_per_epoch=100,
                             metrics=[nDCG@50, nDCG@20, R@50, R@20], # 元コード
                             # metrics=[R@1], # @1は微妙かも
                             val_metric = nDCG@10,
                             early_stopping_patience=200, # ver3で変更
                             gbce_t = 0.75,
                             filter_rated=True,
                             eval_batch_size=512,
                             # recommendation_limit=10, # 元コード
                             recommendation_limit=50,
                             reuse_item_embeddings=False,
                             # 以下が追加
                             num_user=0,
                             train_file_path="",
                             valid_input_path="",
                             valid_output_path="",
                             test_input_path="",
                             test_output_path="",
                             # 以下が追加(ver5)
                             num_items = 0,
                             embedding_dir_train="",
                             embedding_dir_valid="",
                             valid_full_path="",
                             test_full_path=""                      
                             ):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.negs_per_pos = negs_per_pos
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.val_metric = val_metric
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.gbce_t = gbce_t
        self.filter_rated = filter_rated
        self.recommendation_limit = recommendation_limit
        self.eval_batch_size = eval_batch_size
        self.reuse_item_embeddings = reuse_item_embeddings 
        # 以下は追加
        self.num_user = num_user
        self.train_file_path = train_file_path
        self.valid_input_path = valid_input_path
        self.valid_output_path = valid_output_path
        self.test_input_path = test_input_path
        self.test_output_path = test_output_path
        # 以下は追加（ver5）
        self.num_items = num_items
        self.embedding_dir_train = embedding_dir_train
        self.embedding_dir_valid = embedding_dir_valid
        self.valid_full_path = valid_full_path
        self.test_full_path = test_full_path
        
