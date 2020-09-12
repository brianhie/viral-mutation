from Bio import AlignIO
import sys

records = AlignIO.read(sys.argv[1], 'fasta')
with open(sys.argv[2], 'w') as of:
    AlignIO.PhylipIO.PhylipWriter(of).write_alignment(
        records, id_width=100
    )
