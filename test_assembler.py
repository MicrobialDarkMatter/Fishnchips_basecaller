from src.utils.config_loader import load_config
from src.controllers.assembly_controller import AssemblyController
#TEST:
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

pred = new_sequences
config = load_config('./configs/claaudia.json')
controller = AssemblyController(config, 'test_experiment')
controller.assemble(pred[:200], './temps/what.txt')