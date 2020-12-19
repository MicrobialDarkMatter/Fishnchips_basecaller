import numpy as np
import argparse
import shutil
import math
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='Source directory of data to be split into train and validation.')
    parser.add_argument('-t', '--target', required=True, help='Target directory.')
    parser.add_argument('-p', '--train_percent', required=True, help='Percentage of the data that should be allocated for training (int between 0 and 100).')
    return parser.parse_args()

def verify_args(args):
    valid_source = os.path.exists(args.source)
    assert valid_source, 'Source directory is not valid.'
    valid_target = os.path.exists(args.target)
    assert valid_source, 'Target directory is not valid.'
    percentage = int(args.train_percent)
    valid_percentage = 0 <= percentage <= 100
    assert valid_percentage, 'Train percentage is not valid. Value must be an int between 0 and 100'

def verify_split(signals, labels):
    for signal in signals:
        signal_filename = signal.split('.')[0]
        has_matching_label = False
        for label in labels:
            label_filename = label.split('.')[0]
            if signal_filename == label_filename:
                has_matching_label = True  
        assert has_matching_label, f'Signal file {signal_filename} does not correspond to label file.'

def get_source_filenames(source):
    signals = []
    labels = []
    for file in os.listdir(source):
        signals.append(file) if file.endswith('.signal') else labels.append(file)    
    assert len(signals) == len(labels), 'Source directory doesnt have the same amount of signal and label files.'
    assert len(signals) > 0, 'Source directory does not contain relevant data. (Data must have extentions .label or .signal).'
    return signals, labels

def unison_shuffle(lst1, lst2):
    rng_state = np.random.get_state()
    np.random.shuffle(lst1)
    np.random.set_state(rng_state)
    np.random.shuffle(lst2) 
    return lst1, lst2 

def copy_files(filenames, source_dir, target_dir):
    if os.path.isdir(target_dir) == False:
        os.mkdir(target_dir)
    else:
        for filename in os.listdir(target_dir):
            os.unlink(f'{target_dir}/{filename}')
    for filename in filenames:
        shutil.copyfile(f'{source_dir}/{filename}', f'{target_dir}/{filename}')

def main(source_dir, target_dir, train_percentage):
    signals, labels = get_source_filenames(source_dir)
    signals, labels = unison_shuffle(sorted(signals), sorted(labels))
    idx = math.ceil(len(signals) * int(train_percentage) / 100)
    
    train_signals, train_labels = signals[:idx], labels[:idx]
    val_signals, val_labels = signals[idx:], labels[idx:]
    verify_split(train_signals, train_labels)
    verify_split(val_signals, val_labels)
    print(f'Train set - {len(train_signals)} reads.')
    print(f'Validation set - {len(val_signals)} reads.')
    
    print('Copying files...')
    copy_files(train_signals + train_labels, source_dir, f'{target_dir}/train')
    copy_files(val_signals + val_labels, source_dir, f'{target_dir}/validate')

if __name__ == "__main__":
    args = parse_args()
    verify_args(args)
    main(args.source, args.target, args.train_percent)
    print('Script has finished successfully.')