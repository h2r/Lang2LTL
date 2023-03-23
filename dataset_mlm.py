import argparse
from dataset_symbolic import load_split_dataset
from utils import save_to_file

ONE_ARG_SYMBOLS = ['!', 'F', 'G', 'X']
TWO_ARG_SYMBOLS = ['&', '|', 'U', 'M', 'i', 'e']

def mask_1(ltl):
    for op in ONE_ARG_SYMBOLS + TWO_ARG_SYMBOLS:
        ltl = ltl.replace(op, '[mask]')
    return ltl

def construct_dataset(split_dataset_fpath, new_dataset_fpath, mask_func):
    train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)
    new_train_iter = []
    # new_valid_iter = []
    for i in range(len(train_iter)):
        target = f'{train_iter[i][0]}\nLTL: {train_iter[i][1]}'
        input = f'{train_iter[i][0]}\nLTL: {mask_func(train_iter[i][1])}'
        new_train_iter.append((input, target))
    
    # for i in range(len(valid_iter)):
    #     target = f'{valid_iter[i][0]}\nLTL: {valid_iter[i][1]}'
    #     input = f'{valid_iter[i][0]}\nLTL: {mask_func(valid_iter[i][1])}'
    #     new_valid_iter.append((input, target))

    save_to_file({'train_iter': new_train_iter, 'train_meta': train_meta,
                'valid_iter': valid_iter, 'valid_meta': valid_meta},
                new_dataset_fpath)
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dataset_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="complete file path or prefix of file paths to train test split dataset.")
    parser.add_argument("--new_dataset_fpath", type=str)
    args = parser.parse_args()

    construct_dataset(args.split_dataset_fpath, args.new_dataset_fpath, mask_1)