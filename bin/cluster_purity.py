from utils import *

def print_purity(metas, entries):
    for entry in entries:
        tprint('Calculating purity of {}'.format(entry))
        cluster2entry = {}
        for accession in metas:
            meta = metas[accession]
            try:
                cluster = meta['cluster']
            except:
                continue
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

def flu_purity(phylo_method='mafft'):
    from flu import load_meta
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]
    metas = load_meta(meta_fnames)

    if phylo_method == 'mafft':
        cluster_fname = 'target/flu/clusters/all.clusters_0.117.txt'
    elif phylo_method == 'mafft_sl':
        cluster_fname = 'target/flu/clusters/all_singlelink_0.119.txt'
    elif phylo_method == 'clustalomega':
        cluster_fname = 'target/flu/clusters/clustal_omega_clusters_0.382.txt'
    elif phylo_method == 'clustalomega_sl':
        cluster_fname = 'target/flu/clusters/clustal_omega_singlelink_0.3.txt'
    elif phylo_method == 'mrbayes':
        cluster_fname = 'target/flu/clusters/mrbayes_clusters_125.txt'
    elif phylo_method == 'mrbayes_sl':
        cluster_fname = 'target/flu/clusters/mrbayes_singlelink_0.25.txt'
    elif phylo_method == 'raxml':
        cluster_fname = 'target/flu/clusters/raxml_clusters_0.1.txt'
    elif phylo_method == 'raxml_sl':
        cluster_fname = 'target/flu/clusters/raxml_singlelink_0.56.txt'
    elif phylo_method == 'fasttree':
        cluster_fname = 'target/flu/clusters/fasttree_clusters_5.001.txt'
    elif phylo_method == 'fasttree_sl':
        cluster_fname = 'target/flu/clusters/fasttree_singlelink_0.08.txt'
    else:
        raise ValueError('Invalid phylo method {}'.format(phylo_method))

    with open(cluster_fname) as f:
        f.readline()
        for line in f:
            if 'Reference_Perth2009' in line:
                continue
            fields = line.rstrip().split()
            if 'mafft' in phylo_method:
                accession = fields[0].split('_')[2]
                metas[accession]['cluster'] = fields[1]
            else:
                accession = fields[0].split('_')[1]
                metas[accession]['cluster'] = fields[1]

    print_purity(metas, [ 'Subtype', 'Host Species' ])

def hiv_purity(phylo_method='mafft'):
    from hiv import load_meta
    meta_fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]
    metas = load_meta(meta_fnames)
    metas = { accession.split('.')[-1]: metas[accession]
              for accession in metas }

    if phylo_method == 'mafft':
        cluster_fname = 'target/hiv/clusters/all.clusters_0.445.txt'
    elif phylo_method == 'mafft_sl':
        cluster_fname = 'target/hiv/clusters/all_singlelink_0.445.txt'
    elif phylo_method == 'clustalomega':
        cluster_fname = 'target/hiv/clusters/clustal_omega_clusters_0.552.txt'
    elif phylo_method == 'clustalomega_sl':
        cluster_fname = 'target/hiv/clusters/clustal_omega_singelink_0.49.txt'
    elif phylo_method == 'mrbayes':
        cluster_fname = 'target/hiv/clusters/mrbayes_clusters_4.1.txt'
    elif phylo_method == 'mrbayes_sl':
        cluster_fname = 'target/hiv/clusters/mrbayes_singlelink_0.125.txt'
    elif phylo_method == 'raxml':
        cluster_fname = 'target/hiv/clusters/raxml_clusters_0.77.txt'
    elif phylo_method == 'raxml_sl':
        cluster_fname = 'target/hiv/clusters/raxml_singlelink_12.txt'
    elif phylo_method == 'fasttree':
        cluster_fname = 'target/hiv/clusters/fasttree_clusters_1.71.txt'
    elif phylo_method == 'fasttree_sl':
        cluster_fname = 'target/hiv/clusters/fasttree_singlelink_0.64.txt'
    else:
        raise ValueError('Invalid phylo method {}'.format(phylo_method))

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
    flu_purity(sys.argv[1])

    tprint('HIV Env...')
    hiv_purity(sys.argv[1])
