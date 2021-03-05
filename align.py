import mappy as mp
from cigar import create_alignment, print_alignment

with open('./assembly/result.txt', 'r') as f:
    data = f.read()

confidence = data.split('\n')[-1]
consensus = data.split('\n')[-2]
consensus = consensus.upper()
aligner = mp.Aligner('./data/filtered_test_data/Acinetobacter_pittii_16-377-0801/reference.fasta')

try:
    besthit = next(aligner.map(''.join(consensus)))
    cigacc = 1-(besthit.NM/besthit.blen)
    print(cigacc)
except Exception as e:
    print(e)

with open('./data/filtered_test_data/Acinetobacter_pittii_16-377-0801/reference.fasta', 'r') as f:
    reference = f.read()
reference = reference[besthit.r_st:besthit.r_en]
prediction = consensus[besthit.q_st:besthit.q_en]
cigar = besthit.cigar_str

alignment, smdi = create_alignment(reference, prediction, cigar)
prediction_HSP = ''
for e in alignment:
    prediction_HSP += e['prediction']

print(prediction_HSP)