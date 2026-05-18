import spacy
from collections import Counter

from torch import device
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader


def my_tokenizer():
    tokenizer = {}
    de_tokenizer = spacy.load("de_core_news_sm")
    en_tokenizer = spacy.load("en_core_web_sm")

    tokenizer['de'] = (lambda s: [token.text for token in de_tokenizer(s)])
    tokenizer['en'] = (lambda s: [token.text for token in en_tokenizer(s)])
    return tokenizer


class Vocab(object):
    def __init__(self, counter=None, specials=None, min_freq=1):
        if specials is None:
            specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.specials = specials
        self.stoi = {v: k for v, k in zip(specials, range(len(specials)))}
        self.itos = specials[:]  # 如果是self.itos = specials  会导致共用同一个列表 对一个的修改会影响另一个

        for c in counter.most_common():
            if c[1] >= min_freq:
                self.itos.append(c[0])
                self.stoi[c[0]] = len(self.itos) - 1

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.itos[0]))

    def __len__(self):
        return len(self.itos)


def build_vocab(tokenizer, filepath, specials=None, min_freq=1):
    counter = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for string_line in f:
            counter.update(tokenizer(string_line))
    return Vocab(counter, specials, min_freq)


class LoadEnglishGermanDataset():
    def __init__(self, train_file_paths, tokenizer=None, batch_size=16, min_freq=1):
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenizer = tokenizer()  
        self.de_vocab = build_vocab(self.tokenizer['de'], train_file_paths[0], specials,
                                    min_freq) 
        self.en_vocab = build_vocab(self.tokenizer['en'], train_file_paths[1], specials,
                                    min_freq)
        self.PAD_IDX = self.de_vocab['<pad>']
        self.BOS_IDX = self.de_vocab['<bos>']

        self.EOS_IDX = self.de_vocab['<eos>']
        self.batch_size = batch_size

    def data_process(self, filepaths=None):
        raw_de_iter = iter(open(filepaths[0], 'r', encoding='utf-8'))
        raw_en_iter = iter(open(filepaths[1], 'r', encoding='utf-8'))
        data = []
        for (raw_de, raw_en) in tqdm(zip(raw_de_iter, raw_en_iter)):
            de_tensor = torch.tensor([self.de_vocab[token]
                                      for token in self.tokenizer['de'](raw_de.rstrip("\n"))], dtype=torch.long)
            en_tensor = torch.tensor([self.en_vocab[token]
                                      for token in self.tokenizer['en'](raw_en.rstrip("\n"))], dtype=torch.long)
            data.append((de_tensor, en_tensor))

        return data

    def generate_batch(self, data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(de_item)
            en = torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])])
            en_batch.append(en)
        
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)
        

        return de_batch, en_batch

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        val_data = self.data_process(val_file_paths)
        test_data = self.data_process(test_file_paths)

        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, val_iter, test_iter  #[seq_len,batch_size]

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        if mask is not None:
            mask = mask.to(device)
        return mask

    def create_mask(self, src, tgt, device):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_key_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_key_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        # print('=============create_mask里===================')
        # print(src_mask.device, tgt_mask.device,src_key_padding_mask.device,tgt_key_padding_mask.device)
        # print('=============create_mask里===================')
        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask


if __name__ == '__main__':
    counter = Counter()
    data_iter = [
        "hello world",
        'hello from the other side',
        'hello again',
        'world from the other side'
    ]
    for string_line in data_iter:
        counter.update(string_line.split())
    print(counter)
    vocab = Vocab(counter, min_freq=2)
    print(vocab['hello'])
    print(vocab.itos)
    print(vocab.stoi)
    print(vocab['<unk>'])

