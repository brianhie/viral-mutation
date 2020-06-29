from utils import *

def print_purity(metas, entries):
    for entry in entries:
        tprint('Calculating purity of {}'.format(entry))
        cluster2entry = {}
        for accession in metas:
            meta = metas[accession]
            cluster = meta['cluster']
            if cluster not in cluster2entry:
                cluster2entry[cluster] = []
            cluster2entry[cluster].append(meta[entry])
        largest_pct_entry = []
        for cluster in cluster2entry:
            count = Counter(cluster2entry[cluster]).most_common(1)[0][1]
            pct = float(count) / len(cluster2entry[cluster])
            largest_pct_entry.append(pct)
            tprint('Cluster {}, largest % = {}'.format(cluster, pct))
        tprint('Purity, phylo clustering and {}: {}'
               .format(entry, np.mean(largest_pct_entry)))

def flu_purity():
    from flu import load_meta
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]
    metas = load_meta(meta_fnames)

    cluster_fname = 'target/flu/clusters/all.clusters_0.117.txt'
    with open(cluster_fname) as f:
        f.readline()
        for line in f:
            if 'Reference_Perth2009_HA_coding_sequence' in line:
                continue
            fields = line.rstrip().split()
            accession = fields[0].split('_')[2]
            cluster = fields[1]
            metas[accession]['cluster'] = fields[1]

    print_purity(metas, [ 'Subtype', 'Host Species' ])

def hiv_purity():
    from hiv import load_meta
    meta_fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]
    metas = load_meta(meta_fnames)
    metas = { accession.split('.')[-1]: metas[accession]
              for accession in metas }

    cluster_fname = 'target/hiv/clusters/all.clusters_0.445.txt'
    with open(cluster_fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split()
            if fields[0].endswith('NC_001802'):
                accession = 'NC_001802'
            else:
                accession = fields[0].split('_')[-1]
            cluster = fields[1]
            metas[accession]['cluster'] = fields[1]

    print_purity(metas, [ 'subtype' ])

if __name__ == '__main__':
    tprint('Flu HA...')
    flu_purity()

    tprint('HIV Env...')
    hiv_purity()
