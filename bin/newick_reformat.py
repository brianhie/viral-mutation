import sys

newick = ''

with open(sys.argv[1]) as f:
    orig = f.read().rstrip()

orig = orig.replace('(', '(\n').replace(',', '\n,\n').replace(')', ')\n')

for line in orig.split('\n'):
    fields = line.rstrip().split(':')
    if len(fields) == 0:
        continue
    elif len(fields) == 1:
        newick += fields[0]
    else:
        prefix, suffix = fields
        newick += prefix[:30]
        if suffix != '':
            newick += ':'
            newick += suffix

with open(sys.argv[2], 'w') as of:
    of.write(newick + '\n')
