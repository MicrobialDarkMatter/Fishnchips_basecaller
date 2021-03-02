import os
import math
import numpy as np
from Bio.Align.Applications import MafftCommandline
from io import StringIO
from Bio import AlignIO


def assemble(predictions, save_path = ''):
    chunks = get_chunked_alignment(predictions)
    indexes = get_alignment_indexes(chunks)  
    columns = get_columns(chunks, indexes)
    consensus, confidence = get_consensus(columns)
    if save_path != '':
        make_assembly_file(chunks, indexes, consensus, save_path)  
    return consensus, confidence

def print_confidence_string(consesnsus, confidence):
    result = zip(consensus, confidence)
    for char, confidence in result:
        if confidence > 0.9:
            print(f'\033[92m{char}\033[0m', end='')
        elif confidence > 0.75:
            print(f'\033[94m{char}\033[0m', end='')
        elif confidence > 0.5:
            print(f'\u001b[33m{char}\033[0m', end='')
        else:
            print(f'\033[91m{char}\033[0m', end='')

def get_consensus(columns):
    consesnsus = ''
    confidence = []
    for column_str in columns:
        char, score = get_most_frequent_element(column_str)
        consesnsus += char
        confidence.append(score)
    return consesnsus, confidence

def trim_column(str_col):
    m = -1
    n = -1
    for i,char in enumerate(str_col):
        if char in 'actg_':
            m = i
            break
    for i,char in reversed(list(enumerate(str_col))):
        if char in 'actg_':
            n = i
            break
    return str_col[m:n+1]

def get_column_string(chunk, index, col):
    try:
        i = col - index
        arr = np.array(chunk)
        column = arr[:,i]
        s = ''.join(column.tolist())
        return s
    except:
        return '____'

def get_columns(chunks, indexes):
    columns = []
    size = len(chunks[-1][0]) + indexes[-1]
    idx = 0
    for col in range(size):
        next_chunk = col >= len(chunks[idx][0]) + indexes[idx]
        if next_chunk:
            idx += 1
        if idx == len(indexes) -1:
            s = get_column_string(chunks[idx], indexes[idx], col)
            s = trim_column(s)
            columns.append(s)
            continue
        overlap = col >= indexes[idx+1]
        if overlap:
            s1 = get_column_string(chunks[idx], indexes[idx], col)
            s2 = get_column_string(chunks[idx+1], indexes[idx+1], col)
            s = s1 + s2
            s = trim_column(s)
            columns.append(s)            
        else:
            s = get_column_string(chunks[idx], indexes[idx], col)
            s = trim_column(s)
            columns.append(s)
    return columns

def get_alignment_indexes(chunks):
    indexes = []
    indexes.append(0)
    for i,_ in enumerate(chunks):
        if i + 1 == len(chunks):
            break
        c1 = np.array(chunks[i])
        c2 = np.array(chunks[i+1])
        s1 = c1.shape[1]
        s2 = c2.shape[1]
        max_score = -100
        max_idx = None
        for overlap_idx in range(s1):
            score = 0
            for col_idx in range(min(s1 - overlap_idx, s2)):
                c1_col_idx = overlap_idx + col_idx
                c2_col_idx = col_idx
                c1_col = c1[:,c1_col_idx]
                c2_col = c2[:,c2_col_idx]
                e1, _ = get_most_frequent_element(c1_col.tolist())
                e2, _ = get_most_frequent_element(c2_col.tolist())
                score += 1 if e1 == e2 else -1
            if score > max_score:
                max_score = score
                max_idx = overlap_idx   
        indexes.append(max_idx + indexes[-1])
    return indexes

def get_chunked_alignment(predictions, chunk_size=10):
    mafft_path = '/opt/anaconda3/bin/mafft'
    chunks = []
    for i in range(0,len(predictions), chunk_size):
        batch = predictions[i:i+chunk_size]
        temp_filepath = './temps/predications.txt'
        make_prediction_file(batch, temp_filepath)
        mafft_cline = MafftCommandline(mafft_path, input=temp_filepath)
        stdout, _ = mafft_cline()
        alignment = AlignIO.read(StringIO(stdout), 'fasta')
        
        chunk = []
        for line in alignment:
            chunk.append(list(line))
        chunks.append(chunk)
    return chunks

def get_most_frequent_element(lst): 
    assert len(lst) > 0
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(lst): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count and item != '-': 
            count, itm = dict[item], item 
    return (itm), round(count/len(lst), 2)

def make_prediction_file(predictions, path):
    with open(path, 'w') as f:
        for i,prediction in enumerate(predictions):
            f.write(f'>s{i}\n')
            f.write(f'{prediction}\n')

def make_assembly_file(chunks, indexes, consensus, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'a') as f:
        for i,chunk in enumerate(chunks):
            for line in chunk:
                f.write(f'{indexes[i]*"-"}{"".join(line)}\n')
        f.write(consensus)
        
def prep():
    with open('./aligning/sequences.txt', 'r') as f:
        data = f.read()
    sequences = data.split('\n')
    new_sequences = []
    for sequence in sequences[:-1]:
        new_sequence = []
        for char in sequence:
            if char in 'ATCG':
                new_sequence.append(char)
        if len(new_sequence) > 0:
            new_sequences.append(''.join(new_sequence))
    return new_sequences

predictions = prep()
consensus, confidence = assemble(predictions[:1000], './temps/assembly_test.txt')
print_confidence_string(consensus, confidence)