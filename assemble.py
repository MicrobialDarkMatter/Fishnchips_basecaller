from src.controllers.assembly_controller import AssemblyController
from cigar import create_alignment

config = {
    'testing':{
        'mafft':'/opt/anaconda3/bin/mafft'
    }
}

controller = AssemblyController(config, 'dump')
with open('./assembly/predictions.txt', 'r') as f:
    data = f.read()

predications = data.split('\n')
consensus, _ = controller.assemble(predications, './assembly/result.txt')
print(consensus)
print(len(consensus))


