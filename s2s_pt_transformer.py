"""
Based on PyTorch tutorial, Language Translation with nn.transformer and torchtext.
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
import argparse
import os
from pathlib import Path
import math
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter

from dataset_lifted import load_split_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

SRC_LANG, TAR_LANG = 'en', 'ltl'  # 'de', 'en'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS = 3, 3
EMBED_SIZE = 512
NHEAD = 8
DIM_FFN_HID = 512
BATCH_SIZE = 128
NUM_EPOCHS = 128


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size,
                 num_encoder_layers, num_decoder_layers, embed_size, nhead,
                 dim_feedforward, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=embed_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       device=DEVICE)
        self.generator = nn.Linear(embed_size, tar_vocab_size).to(DEVICE)
        self.src_token_embed = TokenEmbedding(src_vocab_size, embed_size).to(DEVICE)
        self.tar_token_embed = TokenEmbedding(tar_vocab_size, embed_size).to(DEVICE)
        self.positional_encoding = PositionalEncoding(embed_size, dropout).to(DEVICE)

    def forward(self, src, tar, src_mask, tar_mask, src_padding_mask, tar_padding_mask,
                memory_key_padding_mask):
        src_embed = self.positional_encoding(self.src_token_embed(src))
        tar_embed = self.positional_encoding(self.tar_token_embed(tar))
        outs = self.transformer(src_embed, tar_embed, src_mask, tar_mask, None,
                                src_padding_mask, tar_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_token_embed(src)), src_mask)

    def decode(self, tar, memory, tar_mask):
        return self.transformer.decoder(self.positional_encoding(self.tar_token_embed(tar)), memory, tar_mask)


class TokenEmbedding(nn.Module):
    """
    Convert tensor of input indices to corresponding tensor of token embeddings.
    """
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):
    """
    Add positional encoding to token embedding to account for word order.
    """
    def __init__(self, embed_size, dropout, max_ntokens=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)
        pos = torch.arange(0, max_ntokens).reshape(max_ntokens, 1)
        pos_embedding = torch.zeros((max_ntokens, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def construct_dataset_meta(train_iter):
    vocab_transform = {}
    tokenizer = get_tokenizer(tokenizer=None)
    for ln in [SRC_LANG, TAR_LANG]:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer, ln),
                                                        min_freq=1,
                                                        specials=SPECIAL_TOKENS,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)  # default index returned when token not found
    src_vocab_size, tar_vocab_size = len(vocab_transform[SRC_LANG]), len(vocab_transform[TAR_LANG])

    text_transform = {
        SRC_LANG: sequential_transforms(tokenizer, vocab_transform[SRC_LANG], tensor_transform),
        TAR_LANG: sequential_transforms(tokenizer, vocab_transform[TAR_LANG], tensor_transform)
    }  # covert raw strings to tensors of indices: tokenize, convert words to indices, add SOS and EOS indices

    return vocab_transform, text_transform, src_vocab_size, tar_vocab_size


def yield_tokens(data_iter, tokenizer, language):
    language_idx = {SRC_LANG: 0, TAR_LANG: 1}
    for sample in data_iter:
        if isinstance(tokenizer, dict):  # if different tokenizers for source and target
            yield tokenizer[language](sample[language_idx[language]])
        else:
            yield tokenizer(sample[language_idx[language]])


def sequential_transforms(*transforms):
    """
    Iteratively apply input transforms to input text.
    """
    def fn(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return fn


def tensor_transform(token_ids):
    """
    Add SOS and EOS indices.
    """
    return torch.cat((torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))


def collate_fn(data_batch):
    src_batch, tar_batch = [], []
    for src_sample, tar_sample in data_batch:
        src_batch.append(text_transform[SRC_LANG](src_sample.rstrip('\n')))
        tar_batch.append(text_transform[TAR_LANG](tar_sample.rstrip('\n')))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tar_batch = pad_sequence(tar_batch, padding_value=PAD_IDX)
    return src_batch, tar_batch


def create_mask(src, tar):
    src_seq_len, tar_seq_len = src.shape[0], tar.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    tar_mask = generate_square_subsequent_mask(tar_seq_len)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tar_padding_mask = (tar == PAD_IDX).transpose(0, 1)
    return src_mask, tar_mask, src_padding_mask, tar_padding_mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train_epoch(model, optimizer, train_iter):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src_batch, tar_batch in train_dataloader:
        src_batch, tar_batch = src_batch.to(DEVICE), tar_batch.to(DEVICE)
        tar_input = tar_batch[:-1, :]
        src_mask, tar_mask, src_padding_mask, tar_padding_mask = create_mask(src_batch, tar_input)

        logits = model(src_batch, tar_input, src_mask, tar_mask, src_padding_mask, tar_padding_mask,
                       src_padding_mask)

        optimizer.zero_grad()
        tar_out = tar_batch[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_dataloader)


def evaluate(model, valid_iter):
    model.eval()
    losses = 0
    valid_dataloader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src_batch, tar_batch in valid_dataloader:
        src_batch, tar_batch = src_batch.to(DEVICE), tar_batch.to(DEVICE)
        tar_input = tar_batch[:-1, :]
        src_mask, tar_mask, src_padding_mask, tar_padding_mask = create_mask(src_batch, tar_input)

        logits = model(src_batch, tar_input, src_mask, tar_mask, src_padding_mask, tar_padding_mask,
                       src_padding_mask)

        tar_out = tar_batch[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_dataloader)


def translate(model, vocab_transform, text_transform, src_sentence):
    model.eval()
    src = text_transform[SRC_LANG](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    tar_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens+5, start_symbol=SOS_IDX).flatten()
    return " ".join(vocab_transform[TAR_LANG].lookup_tokens(list(tar_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src, src_mask = src.to(DEVICE), src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for _ in range(max_len-1):
        memory = memory.to(DEVICE)
        tar_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)
        out = model.decode(ys, memory, tar_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dataset_fpath', type=str, default='data/split_symbolic_no_perm_batch1_ltl_type_2_42.pkl',
                        help='complete file path or prefix of file paths to train and test data for supervised seq2seq')
    args = parser.parse_args()

    if "pkl" in args.split_dataset_fpath:  # complete file path, e.g. data/split_symbolic_no_perm_batch1_utt_0.2_42.pkl
        split_dataset_fpaths = [args.split_dataset_fpath]
    else:  # prefix of file paths, e.g. split_symbolic_no_perm_batch1_utt
        split_dataset_fpaths = [os.path.join("data", fpath) for fpath in os.listdir("data") if args.split_dataset_fpath in fpath]

    for split_dataset_fpath in split_dataset_fpaths:
        # Load train, test data
        train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)
        vocab_transform, text_transform, SRC_VOCAB_SIZE, TAR_VOCAB_SIZE = construct_dataset_meta(train_iter)

        # Train and save model
        transformer = Seq2SeqTransformer(SRC_VOCAB_SIZE, TAR_VOCAB_SIZE,
                                         NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                         DIM_FFN_HID)
        for param in transformer.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        transformer = transformer.to(DEVICE)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        writer = SummaryWriter()  # writer will output to ./runs/ directory by default; activate: tensorboard --logdir=runs

        for epoch in range(1, NUM_EPOCHS+1):
            start_time = timer()
            train_loss = train_epoch(transformer, optimizer, train_iter)
            end_time = timer()
            valid_loss = evaluate(transformer, valid_iter)
            print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {valid_loss:.3f}\n'
                  f'Epoch time: {(end_time-start_time):.3f}s')
            writer.add_scalars("Train Loss", {"train_loss": train_loss, "valid_loss": valid_loss}, epoch)
            model_fpath = f'model/s2s_pt_transformer_{Path(split_dataset_fpath).stem}_epoch{epoch}.pth'
            torch.save(transformer.state_dict(), model_fpath)
        writer.flush()
        writer.close()
