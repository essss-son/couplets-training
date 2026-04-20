import logging
import os

import torch


class Config():
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_corpus_file_paths = [os.path.join(self.dataset_dir, 'train.de'),
                                        os.path.join(self.dataset_dir, 'train.en')]
        self.val_corpus_file_paths = [os.path.join(self.dataset_dir, 'val.de'),
                                      os.path.join(self.dataset_dir, 'val.en')]
        self.test_corpus_file_paths = [os.path.join(self.dataset_dir, 'test_2016_flickr.de'),
                                       os.path.join(self.dataset_dir, 'test_2016_flickr.en')]
        self.min_freq = 2

        # 模型相关配置
        self.batch_size = 64
        self.d_model = 512
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 1e-8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 500
        self.model_save_path = os.path.join(self.project_dir, 'cache')
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        # logger_init(log_file_name='log_train',
        #             log_level=logging.DEBUG,
        #             log_dir=self.model_save_path)


if __name__ == '__main__':
    print(__file__)
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(torch.cuda.is_available())