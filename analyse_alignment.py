import os
import json
from Bio import SeqIO
from src.controllers.file_controller import FileController

def get_prediction_filepath(read_id, iteration, bacteria, experiment_name):
    file_controller = FileController(experiment_name)
    filepath = file_controller.get_prediction_filepath(read_id, iteration, bacteria)
    assert os.path.exists(filepath), ' ! Prediction file does not exist'
    return filepath
    
def load_fasta_sequence(filename):
    fasta_sequences = SeqIO.parse(open(filename),'fasta')
    return str(next(fasta_sequences).seq)

def get_referece_filepath(bacteria):
    return f'./data/filtered_test_data/{bacteria}/reference.fasta'

def load_cigar_string(read_id, experiment_name):
    file_controller = FileController(experiment_name)
    path = file_controller.get_testing_filepath()
    with open(path, 'r') as f:
        results = json.load(f)
    for result in results:
        if result['read_id'] == read_id:
            return result['cig'], result['r_st'], result['r_en']
    raise ' ! Cigar string not found.'

experiment_name = 'experiment_claaudia_7_window'
bacteria = 'Acinetobacter_pittii_16-377-0801'
read_id = 'f5312fa6-4a40-4b19-910c-70c814d551b5'
iteration = '0'

prediction_filepath = get_prediction_filepath(read_id, iteration, bacteria, experiment_name)
reference_filepath = get_referece_filepath(bacteria)

prediction = load_fasta_sequence(prediction_filepath)
reference = load_fasta_sequence(reference_filepath)
cigar, s_idx, e_idx = load_cigar_string(read_id, experiment_name)

aligned_reference = reference[s_idx:e_idx]


# print(len(prediction))
# print(len(reference))
# print(len(cigar))

print(prediction[:20])
print(aligned_reference[:20])
print(cigar[:20])
