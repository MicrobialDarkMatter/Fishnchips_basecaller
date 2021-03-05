import re
import os
import json
from Bio import SeqIO
from src.controllers.file_controller import FileController

class bcolors:
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    END = '\033[0m'
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
            return result['cig'], result['r_st'], result['r_en'], result['q_st'], result['q_en'], result['blen'], result['NM']
    raise ' ! Cigar string not found.'

def create_alignment(reference, prediction, cigar):
    cigar_list = re.findall(r'[\d]+[SMDI]', cigar)
    alignment = []
    smdi_dict = {"S":0,"M":0,"D":0,"I":0}
    for operation in cigar_list:
        prediction_out = ''
        reference_out = ''
        operation_type = operation[-1]
        operation_count = int(operation[:-1])
        if operation_type == 'M':
            for idx in range(operation_count):
                if prediction[idx] == reference[idx]:
                    smdi_dict['M'] += 1
                    prediction_out += f'{bcolors.GREEN}{prediction[idx]}{bcolors.END}'
                    # prediction_out += prediction[idx]
                else:
                    smdi_dict['S'] += 1
                    prediction_out += f'{bcolors.FAIL}{prediction[idx]}{bcolors.END}'
                    # prediction_out += prediction[idx]
            reference_out = reference[:operation_count]
            prediction = prediction[operation_count:]
            reference = reference[operation_count:]
        elif operation_type == 'D':
            prediction_out = f'{bcolors.FAIL}{operation_count * "_"}{bcolors.END}'
            # prediction_out = operation_count * "_"
            reference_out = reference[:operation_count]
            reference = reference[operation_count:]
            smdi_dict['D'] += operation_count
        elif operation_type == 'I':
            prediction_out = f'{bcolors.FAIL}{prediction[:operation_count]}{bcolors.END}'
            # prediction_out = prediction[:operation_count]
            reference_out = operation_count * "_"
            prediction = prediction[operation_count:]
            smdi_dict['I'] += operation_count
        else:
            print(operation_type)
            raise 'New operation type! can this be the missing substitution?'

        alignment.append({
            'prediction':prediction_out,
            'reference':reference_out
        })

    return alignment, smdi_dict

def print_alignment(alignment):
    prediction_str = ''
    reference_str = ''
    for i,e in enumerate(alignment):
        if i % 10 == 0:
            print(prediction_str)
            print(reference_str)
            print()
            prediction_str = ''
            reference_str = ''
        prediction_str += e['prediction']
        reference_str += e['reference']

# experiment_name = 'experiment_claaudia_7_window_alignment'
# bacteria = 'Haemophilus_haemolyticus_M1C132_1'
# read_id = '77d998e9-a805-4515-9ada-eaf3bf200d5c'
# iteration = '1'
# expected_acc = 0.7127824736932896

# # experiment_name = 'experiment_claaudia_7_window_alignment'
# # bacteria = 'Acinetobacter_pittii_16-377-0801'
# # read_id = 'f5312fa6-4a40-4b19-910c-70c814d551b5'
# # iteration = '0'
# # expected_acc = 0.7683668925938753

# prediction_filepath = get_prediction_filepath(read_id, iteration, bacteria, experiment_name)
# reference_filepath = get_referece_filepath(bacteria)

# prediction = load_fasta_sequence(prediction_filepath)
# reference = load_fasta_sequence(reference_filepath)
# cigar, rs_idx, re_idx, ps_idx, pe_idx, blen, nm = load_cigar_string(read_id, experiment_name)

# aligned_reference = reference[rs_idx:re_idx]
# aligned_predition = prediction[ps_idx:pe_idx]

# alignment, smdi_dict = create_alignment(aligned_reference, aligned_predition, cigar)
# # print_alignment(alignment)

# prediction_HSP = ''
# reference_HSP = ''
# for e in alignment:
#     prediction_HSP += e['prediction']
#     reference_HSP += e['reference']

# matches = 0
# length = len(prediction_HSP)
# for i,_ in enumerate(prediction_HSP):
#     if prediction_HSP[i] == reference_HSP[i]:
#         matches += 1
# pident = matches / length
# print(pident)
# print(expected_acc)


# print(smdi_dict)
# # s = smdi_dict['S']
# # d = smdi_dict['D']
# # i = smdi_dict['I']
# m = smdi_dict['M']

# # r = 1- (nm/blen)
# # r2 = 1- (m / (m+i+d+s))

# # print(m)
# # print(nm)

# # print(f'Expected: 0.7127824736932896')
# # print(f'Actual MAPPY:{r}')
# # print(f'Actual Manual:{r2}')

# # r = m / (m+s+i+d)
# # print(r)
# # print(blen)
# # print(re_idx - rs_idx)
# # print(pe_idx - ps_idx)

# # print(f'Matches: {m/(s+i+d+m)} %')
# # print(f'No S: {m/(i+d+m)} %')

