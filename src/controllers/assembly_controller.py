import os
import numpy as np
from io import StringIO
from Bio import AlignIO
from Bio.Align.Applications import MafftCommandline

class AssemblyController():
    def __init__(self, config, experiment_name):
        self.mafft_path = config['testing']['mafft']
        self.experiment_name = experiment_name
    
    def assemble(self, predictions, filepath):
        chunks = self.align_predictions(predictions)
        indexes = self.align_chunks(chunks)
        columns = self.get_alignment_columns(chunks, indexes)
        consensus, confidence = self.get_consensus(columns)
        confidence_str = self.get_confidence_string(confidence)
        self.save(chunks, indexes, consensus, confidence_str, filepath)
        return consensus, confidence

    def print_assembly_string(consesnsus, confidence):
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

    def align_predictions(self, predictions):
        chunks = []
        chunk_size = 10
        for i in range(0,len(predictions), chunk_size):
            batch = predictions[i:i+chunk_size]
            temp_filepath = f'./trained_models/{self.experiment_name}/temp_predictions.txt'
            self.create_mafft_input_file(batch, temp_filepath)
            mafft_cline = MafftCommandline(self.mafft_path, input=temp_filepath)
            stdout, _ = mafft_cline()
            alignment = AlignIO.read(StringIO(stdout), 'fasta')
            chunk = []
            for line in alignment:
                chunk.append(list(line))
            chunks.append(chunk)
        return chunks

    def align_chunks(self, chunks):
        indexes = []
        indexes.append(0)
        for i,_ in enumerate(chunks):
            if i + 1 == len(chunks):
                break
            c1 = np.array(chunks[i])
            c2 = np.array(chunks[i+1])
            s1 = c1.shape[1]
            s2 = c2.shape[1]
            max_score = -1e5
            max_idx = 0
            for overlap_idx in range(s1//2,s1):
                score = 0
                for col_idx in range(min(s1 - overlap_idx, s2)):
                    c1_col_idx = overlap_idx + col_idx
                    c2_col_idx = col_idx
                    c1_col = c1[:,c1_col_idx]
                    c2_col = c2[:,c2_col_idx]
                    e1, _ = self.get_most_frequent_element(c1_col.tolist())
                    e2, _ = self.get_most_frequent_element(c2_col.tolist())
                    score += 1 if e1 == e2 else -1
                if score > max_score:
                    max_score = score
                    max_idx = overlap_idx   
            indexes.append(max_idx + indexes[-1])
        return indexes
    
    def get_alignment_columns(self, chunks, indexes):
        columns = []
        size = len(chunks[-1][0]) + indexes[-1]
        idx = 0
        for col in range(size):
            next_chunk = col >= len(chunks[idx][0]) + indexes[idx]
            if next_chunk:
                idx += 1
            if idx == len(indexes) -1:
                s = self.get_column_as_string(chunks[idx], indexes[idx], col)
                s = self.trim_column(s)
                columns.append(s)
                continue
            overlap = col >= indexes[idx+1]
            if overlap:
                s1 = self.get_column_as_string(chunks[idx], indexes[idx], col)
                s2 = self.get_column_as_string(chunks[idx+1], indexes[idx+1], col)
                s = s1 + s2
                s = self.trim_column(s)
                columns.append(s)            
            else:
                s = self.get_column_as_string(chunks[idx], indexes[idx], col)
                s = self.trim_column(s)
                columns.append(s)
        return columns   

    def get_consensus(self, columns):
        consesnsus = ''
        confidence = []
        for column in columns:
            char, score = self.get_most_frequent_element(column)
            consesnsus += char
            confidence.append(score)
        return consesnsus, confidence

    def save(self, chunks, indexes, consensus, confidence, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, 'a') as f:
            for i,chunk in enumerate(chunks):
                for line in chunk:
                    f.write(f'{indexes[i]*"-"}{"".join(line)}\n')
            f.write(f'{consensus}\n')
            f.write(confidence)

    def get_confidence_string(self, confidence):
        result = ''
        for value in confidence:
            if value >= 0.9:
                result += '*'
            elif value >= 0.75:
                result += '-'
            else:
                result += '_'
        return result

    def get_column_as_string(self, chunk, index, col):
        try:
            i = col - index
            arr = np.array(chunk)
            column = arr[:,i]
            s = ''.join(column.tolist())
            return s
        except:
            return '----'

    def trim_column(self, str_column):
        m = -1
        n = -1
        for i,char in enumerate(str_column):
            if char in 'actg':
                m = i
                break
        for i,char in reversed(list(enumerate(str_column))):
            if char in 'actg':
                n = i
                break
        return str_column[m:n+1]

    def create_mafft_input_file(self, predictions, path):
        with open(path, 'w') as f:
            for i,prediction in enumerate(predictions):
                f.write(f'>s{i}\n')
                f.write(f'{prediction}\n')

    def get_most_frequent_element(self, lst):
        if len(lst) == 0:
            return '-', 0
        dict = {} 
        count, itm = 0, '' 
        for item in reversed(lst): 
            dict[item] = dict.get(item, 0) + 1
            if dict[item] >= count and item != '-': 
                count, itm = dict[item], item 
        return (itm), round(count/len(lst), 2)     