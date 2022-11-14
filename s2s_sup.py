import argparse
import torch

from s2s_transformer import Seq2SeqTransformer, \
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID
from s2s_transformer import translate as transformer_translate
from s2s_transformer import construct_dataset as transformer_construct_dataset


class Seq2Seq:
    """
    Model inference.
    """
    def __init__(self, vocab_transform, text_transform, src_vocab_sz, tar_vocab_sz, fpath_load, model_type='transformer'):
        if model_type == 'transformer':
            self.model = Seq2SeqTransformer(src_vocab_sz, tar_vocab_sz,
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
            self.model_translate = transformer_translate
            self.vocab_transform = vocab_transform
            self.text_transform = text_transform
        else:
            raise ValueError(f'ERROR: Model type not recognized: {model_type}')

        self.model.load_state_dict(torch.load(fpath_load))

    def translate(self, query):
        return self.model_translate(self.model, self.vocab_transform, self.text_transform, query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/symbolic_pairs.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default='model/s2s_transformer.pth', help='file path to trained supervised seq2seq model')
    args = parser.parse_args()

    _, _, vocab_transform, text_transform, src_vocab_size, tar_vocab_size = transformer_construct_dataset(args.data)
    s2s_transformer = Seq2Seq(vocab_transform, text_transform, src_vocab_size, tar_vocab_size, args.model)
    print(s2s_transformer.translate("go to A then to B"))
