import math
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

SRC_LANG, TAR_LANG = 'de', 'en'  # 'en', 'ltl'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS = 3, 3
EMBED_SIZE = 512
NHEAD = 8
DIM_FFN_HID = 512
BATCH_SIZE = 128
NUM_EPOCHS = 18


class Seq2Seq:
    """
    Inference trained model.
    """
    def __init__(self, src_vocab_sz, tar_vocab_sz, model_type='transformer', fpath_load='s2s_transformer.pth'):
        if model_type == 'transformer':
            self.model = Seq2SeqTransformer(src_vocab_sz, tar_vocab_sz,
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
        else:
            raise ValueError(f'ERROR: Model type not recognized: {model_type}')

        self.model.load_state_dict(torch.load(fpath_load))

    def translate(self, query):
        return self.model(query)


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
        self.register_buff('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


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
                                       dropout=dropout)
        self.generator = nn.Linear(embed_size, tar_vocab_size)
        self.src_token_embed = TokenEmbedding(src_vocab_size, embed_size)
        self.tar_token_embed = TokenEmbedding(tar_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout)

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


def yield_tokens(data_iter, language):
    language_idx = {SRC_LANG: 0, TAR_LANG: 1}
    for sample in data_iter:
        yield token_transform[language](sample[language_idx[language]])


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tar):
    src_seq_len, tar_seq_len = src.shape[0], tar.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    tar_mask = generate_square_subsequent_mask(tar_seq_len)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tar_padding_mask = (tar == PAD_IDX).transpose(0, 1)
    return src_mask, tar_mask, src_padding_mask, tar_padding_mask


def tensor_transform(token_ids):
    """
    Add SOS and EOS.
    """
    return torch.cat((torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))


def sequential_transforms(*transforms):
    def fn(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)  # TODO: output set by only last iteration
        return txt_input
    return fn


def collate_fn(data_batch):
    src_batch, tar_batch = [], []
    for src_sample, tar_sample in data_batch:
        src_batch.append(text_transform[SRC_LANG](src_sample.rstrip('\n')))
        tar_batch.append(text_transform[TAR_LANG](tar_sample.rstrip('\n')))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tar_batch = pad_sequence(tar_batch, padding_value=PAD_IDX)
    return src_batch, tar_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANG, TAR_LANG))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src_batch, tar_batch in train_dataloader:
        src_batch, tar_batch = src_batch.to(DEVICE), tar_batch.to(DEVICE)

        tar_input = tar_batch[:-1, :]
        src_mask, tar_mask, src_padding_mask, tar_padding_mask = create_mask(src_batch, tar_input)

        logits = model(src_batch, tar_input, src_mask, tar_mask, src_padding_mask, tar_padding_mask,
                       src_padding_mask)

        optimizer.zero_grad()
        tar_out = tar_batch[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_out.shape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANG, TAR_LANG))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src_batch, tar_batch in val_dataloader:
        src_batch, tar_batch = src_batch.to(DEVICE), tar_batch.to(DEVICE)

        tar_input = tar_batch[:-1, :]
        src_mask, tar_mask, src_padding_mask, tar_padding_mask = create_mask(src_batch, tar_input)

        logits = model(src_batch, tar_input, src_mask, tar_mask, src_padding_mask, tar_padding_mask,
                       src_padding_mask)

        tar_out = tar_batch[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


if __name__ == '__main__':
    token_transform = {
        SRC_LANG: get_tokenizer('spacy', language='de_core_news_sm'),
        TAR_LANG: get_tokenizer('spacy', language='en_core_web_sm')
    }

    vocab_transform = {}
    multi30k.URL['train'] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL['valid'] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    for ln in [SRC_LANG, TAR_LANG]:
        train_iter = Multi30k(split='train', language_pair=(SRC_LANG, TAR_LANG))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=SPECIAL_SYMBOLS,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)  # default index returned when token not found

    text_transform = {
        SRC_LANG: sequential_transforms(token_transform[SRC_LANG], vocab_transform[SRC_LANG], tensor_transform),
        TAR_LANG: sequential_transforms(token_transform[TAR_LANG], token_transform[SRC_LANG], tensor_transform)
    }

    breakpoint()

    SRC_VOCAB_SIZE, TAR_VOCAB_SIZE = len(vocab_transform[SRC_LANG]), len(vocab_transform[TAR_LANG])

    transformer = Seq2SeqTransformer(SRC_VOCAB_SIZE, TAR_VOCAB_SIZE,
                                     NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                     DIM_FFN_HID)

    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}\n'
              f'Epoch time: {(end_time-start_time):.3f}s')
