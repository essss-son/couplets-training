import sys

sys.path.append('../')

from utils import LoadEnglishGermanDataset
from config import Config
from utils import my_tokenizer

if __name__ == '__main__':
    config = Config()
    dataloader = LoadEnglishGermanDataset(config.test_corpus_file_paths, tokenizer=my_tokenizer,
                                          batch_size=config.batch_size, min_freq=config.min_freq)
    train_iter, valid_inter, test_iter = dataloader.load_train_val_test_data(config.test_corpus_file_paths,
                                                                             config.test_corpus_file_paths,
                                                                             config.test_corpus_file_paths)
    print(dataloader.PAD_IDX)

    for src, tgt in train_iter:
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = dataloader.create_mask(src, tgt_input)

        print("src shape: ", src.shape)
        print(src.transpose(0, 1)[:3])
        print("tgt shape: ", tgt.shape)
        print('src input shape: ', src.shape)
        print("src_padding_mask shape: (batch_size,src_len)", src_padding_mask.shape)
        print("tgt input shape: ", tgt_input.shape)
        print(tgt_input.transpose(0, 1)[:3])
        print('tgt_padding_mask shape: (batch_size,tgt_len)', tgt_padding_mask.shape)
        print("tgt output shape: ", tgt_output.shape)
        print(tgt_output.transpose(0, 1)[:3])
        print("tgt_mask shape (tgt_len,tgt_len)", tgt_mask.shape)
        print('==========================================')
        break
