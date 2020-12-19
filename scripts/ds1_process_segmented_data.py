import numpy as np
import argparse
import h5py
import os

PADDING_TOKEN = 0
START_TOKEN = 5 
END_TOKEN = 6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='Source filepath of labelled segmented data, preprocessed by chiron.')
    parser.add_argument('-t', '--target', required=True, help='Target filepath.')
    args = parser.parse_args()
    verify_args(args)
    return args

def verify_args(args):
    valid_source = os.path.exists(args.source)
    assert valid_source, 'Source filepath is not valid.'
    valid_target = os.path.exists(args.target)
    assert valid_source, 'Target filepath is not valid.'

def load_file(source_filepath):
    with h5py.File(source_filepath, 'r') as f:
        x = f['event']['record'][:]
        y = f['label']['record'][:]
    return x,y

def save_file(target_filepath, x, y):
    if os.path.isfile(target_filepath):
        os.unlink(target_filepath)
    with h5py.File(target_filepath, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)

def alter_labels(y):
    y_fnc = np.zeros(shape=(y.shape[0], y.shape[1]+2)) #+ space for start and end tokens
       
    for i, y_row in enumerate(y):
        print(f'Processing: {i+1}/{y.shape[0]}', end='\r')
        y_fnc_row = np.vectorize(lambda x: x+1)(y_row)
        y_fnc_row = np.append(y_row, [PADDING_TOKEN])
        y_fnc_row = np.insert(y_fnc_row, 0, START_TOKEN, axis=0)
        y_fnc_row[np.argmax(y_fnc_row==0)] = END_TOKEN
        y_fnc[i] = y_fnc_row
    print()
    return y_fnc

def verify_labels(y, y_fnc):
    assert y_fnc.shape == (y.shape[0], y.shape[1]+2)
    assert sum(y_fnc[0]) == sum(y[0]) + START_TOKEN + END_TOKEN
    assert sum(y_fnc[-1]) == sum(y[-1]) + START_TOKEN + END_TOKEN

def main(source_filepath, target_filepath):
    x,y = load_file(source_filepath)
    y_fnc = alter_labels(y)
    verify_labels(y, y_fnc)
    save_file(target_filepath, x, y_fnc)

if __name__ == "__main__":
    args = parse_args()
    main(args.source, args.target)
    print('Script has finished successfully.')