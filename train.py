import torch
from config import Config
from utils import LoadEnglishGermanDataset, my_tokenizer
import os
import logging
from model import Translation_model
from model import CustomSchedule
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,  # 设置最低显示级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# logits:[tgt_len, batch_size, tgt_vocab_size]  y_true:[tgt_len,batch_size]
def accuracy(logits, y_true, PAD_IDX):
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    y_true = y_true.transpose(0, 1).reshape(-1)
    acc = y_pred.eq(y_true)
    mask = torch.logical_not(y_true.eq(PAD_IDX))
    acc = acc.logical_and(mask)
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


def train_model(config=None):
    data_loader = LoadEnglishGermanDataset(
        config.train_corpus_file_paths,
        tokenizer=my_tokenizer,
        batch_size=config.batch_size,
        min_freq=config.min_freq
    )

    train_iter, val_iter, test_iter = data_loader.load_train_val_test_data(
        train_file_paths=config.train_corpus_file_paths,
        val_file_paths=config.val_corpus_file_paths,
        test_file_paths=config.test_corpus_file_paths
    )

    model = Translation_model(
        src_vocab_size=len(data_loader.de_vocab),
        tgt_vocab_size=len(data_loader.en_vocab),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    ).to(config.device)

    model_save_path = os.path.join(config.model_save_path, 'model.pt')

    if os.path.exists(model_save_path):
        loaded_params = torch.load(model_save_path, map_location=config.device)
        model.load_state_dict(loaded_params)
        logging.info(f'Loaded existed model from {model_save_path}')

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=data_loader.PAD_IDX
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )

    lr_scheduler = CustomSchedule(
        config.d_model,
        optimizer=optimizer
    )

    best_val_acc = 0.0

     for epoch in range(config.epochs):

        model.train()

        total_loss = 0.0
        total_acc = 0.0

        for idx, (src, tgt) in enumerate(train_iter):

            src = src.to(config.device)  # [src_len,batch_size]
            tgt = tgt.to(config.device)  # [tgt_len,batch_size]

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_loader.create_mask(
                src,
                tgt_input,
                config.device
            )

            logits = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=None,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            # logits:[tgt_len, batch_size, tgt_vocab_size]  tgt_output:[tgt_len,batch_size]
            # 为了符合loss的输入格式 对logits和tgt_output进行view   但是会在调用accuracy时报错 原因是只需要在loss里面保持view后的维度就行
            # loss计算之后的维度仍需保持为logits:[tgt_len, batch_size, tgt_vocab_size]  tgt_output:[tgt_len,batch_size] 然后传入accuracy  所以我把以下两行注释掉
            # logits = logits.view(-1, logits.size(-1))
            # tgt_output = tgt_output.view(-1)

            optimizer.zero_grad()
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            loss.backward()
            lr_scheduler.step()  ###############
            optimizer.step()
            losses += loss.item()
            # logits:[tgt_len, batch_size, tgt_vocab_size]  tgt_output:[tgt_len,batch_size]
            acc, _, _ = accuracy(logits, tgt_output, data_loader.PAD_IDX)
            msg = f'Epoch:{epoch}, Batch:[{idx}/{len(train_iter)}], Train loss: {loss.item():.3f}, Train acc: {acc:.3f}'
            logging.info(msg)
        train_loss = losses / len(train_iter)
        msg = f'Epoch:{epoch},  Train loss: {train_loss:.3f}'
        logging.info(msg)
        if epoch % 2 == 0:
            acc = evaluate(config, val_iter, translation_model, data_loader)
            logging.info(f"Accuracy on Validation set: {acc:}")
            state_dict = deepcopy(translation_model.state_dict())
            torch.save(state_dict, model_save_path)


def evaluate(config, val_iter, model, data_loader):
    model.eval()
    correct, totals = 0, 0
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(val_iter):
            src = src.to(config.device)  # [src_len,batch_size]
            tgt = tgt.to(config.device)  # [tgt_len,batch_size]
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_loader.create_mask(src, tgt_input,
                                                                                             config.device)

            logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None,
                           src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            _, c, t = accuracy(logits, tgt_output, data_loader.PAD_IDX)
            correct += c
            totals += t
    model.train()
    return float(correct) / totals


if __name__ == '__main__':
    config = Config()
    train_model(config)
