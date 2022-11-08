import torch

from s2s_transformer import Seq2SeqTransformer, translate,\
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID


class Seq2Seq:
    """
    Model inference.
    """
    def __init__(self, src_vocab_sz, tar_vocab_sz, fpath_load, model_type='transformer'):
        if model_type == 'transformer':
            self.model = Seq2SeqTransformer(src_vocab_sz, tar_vocab_sz,
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
        else:
            raise ValueError(f'ERROR: Model type not recognized: {model_type}')

        self.model.load_state_dict(torch.load(fpath_load))

    def translate(self, query):
        return translate(self.model, query)
