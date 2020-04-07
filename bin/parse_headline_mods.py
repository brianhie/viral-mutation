from utils import *

from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
from nltk.corpus import wordnet as wn
from pattern3.en import conjugate, singularize

info_prefix = 'Original headline: '

def extract_content(line):
    return (': '.join(' | '.join(line.split(' | ')[1:])
                      .strip().split(': ')[:-1]))

def read_headline_info(f, orig_line):
    orig = orig_line[len(info_prefix):]

    info = { 'orig': orig }

    assert(f.readline().rstrip().endswith('Modifications:'))

    info['mod'] = extract_content(f.readline())
    info['mod_1'] = extract_content(f.readline())
    info['mod_2'] = extract_content(f.readline())

    assert(f.readline().rstrip().endswith('Least change:'))

    info['least_0'] = extract_content(f.readline())
    info['least'] = extract_content(f.readline())
    info['least_1'] = extract_content(f.readline())

    assert(f.readline().rstrip().endswith('Most change:'))

    info['most'] = extract_content(f.readline())
    info['most_1'] = extract_content(f.readline())
    info['most_2'] = extract_content(f.readline())

    return info

def find_diff(headline1, headline2):
    for idx, (word1, word2) in enumerate(
            zip(headline1.split(' '), headline2.split(' '))):
        if word1 != word2:
            return idx

flair_pos = None
def pos_tag(sentence, backend='nltk'):
    global flair_pos

    if backend == 'nltk':
        return nltk.pos_tag(sentence.split(' '))

    elif backend == 'flair':
        if flair_pos is None:
            flair_pos = SequenceTagger.load('pos')
        sentence_info = Sentence(sentence)
        flair_pos.predict(sentence_info)
        tagged = sentence_info.to_tagged_string().split(' ')
        assert(len(tagged) % 2 == 0)
        parsed = []
        for i in range(len(tagged) // 2):
            idx = i * 2
            tag = tagged[idx + 1]
            assert(tag.startswith('<') and tag.endswith('>'))
            parsed.append((tagged[idx], tag))
        return parsed

    else:
        raise ValueError('Invalid backend: {}'.format(backend))

def find_semantic_distance(wordtag1, wordtag2):
    word1, tag1 = wordtag1
    word2, tag2 = wordtag2

    if tag1.startswith('NN') and tag2.startswith('NN'):
        if tag1 == 'NNS':
            word1 = singularize(word1)
        if tag2 == 'NNS':
            word2 = singularize(word2)
        try:
            syn1 = wn.synset('{}.n.01'.format(word1))
            syn2 = wn.synset('{}.n.01'.format(word2))
        except nltk.corpus.reader.wordnet.WordNetError:
            return None
        return syn1.path_similarity(syn2), syn1.wup_similarity(syn2)

    if tag1.startswith('VB') and tag2.startswith('VB'):
        try:
            word1 = conjugate(word1, 'inf')
        except RuntimeError:
            pass
        try:
            word2 = conjugate(word2, 'inf')
        except RuntimeError:
            pass
        try:
            syn1 = wn.synset('{}.v.01'.format(word1))
            syn2 = wn.synset('{}.v.01'.format(word2))
        except nltk.corpus.reader.wordnet.WordNetError:
            return None
        return syn1.path_similarity(syn2), syn1.wup_similarity(syn2)

    return None

def part_of_speech(infos, backend='nltk', n_most=10):
    categories = [ 'mod', 'least', 'most' ]

    pos_diff_change = { category: [] for category in categories }
    n_pos_change = { category: [] for category in categories }
    pct_pos_change = { category: [] for category in categories }

    if backend == 'nltk':
        wordnet_path_dist = { category: [] for category in categories }
        wordnet_wup_dist = { category: [] for category in categories }

    for info in infos:
        for category in categories:
            diff_idx = find_diff(info['orig'], info[category])

            pos_orig = pos_tag(info['orig'], backend=backend)
            pos_category = pos_tag(info[category], backend=backend)
            assert(len(pos_orig) == len(pos_category))

            word_orig = pos_orig[diff_idx]
            word_cat = pos_category[diff_idx]

            if (word_orig[1].startswith('NN') and word_cat[1].startswith('NN')) or \
               (word_orig[1].startswith('VB') and word_cat[1].startswith('VB')):
                pos_diff_change[category].append((word_orig, word_cat))

            n_pos_change[category].append(sum([
                pos_orig[i][1] != pos_category[i][1]
                for i in range(len(pos_orig))
            ]))
            pct_pos_change[category].append(sum([
                pos_orig[i][1] != pos_category[i][1]
                for i in range(len(pos_orig))
            ]) / float(len(pos_orig)) * 100.)

            if backend == 'nltk':
                dists = find_semantic_distance(word_orig, word_cat)
                if dists is not None:
                    path_dist, wup_dist = dists
                    wordnet_path_dist[category].append(path_dist)
                    wordnet_wup_dist[category].append(wup_dist)

    for category in categories:
        print('Category: {}'.format(category))

        print('\tMost common word/POS changes, {}:'.format(category))
        for result, count in Counter(
                pos_diff_change[category]).most_common(n_most):
            print('\t{}: {}'.format(result, count))

        change_distr = np.array(n_pos_change[category])
        print('\tNumber of POS changes, mean: {:.4f}, median: {}, '
              'min: {}, max: {}, std: {:.4f}'
              .format(np.mean(change_distr), np.median(change_distr),
                      change_distr.min(), change_distr.max(),
                      change_distr.std()))

        change_distr = np.array(pct_pos_change[category])
        print('\tPercentage of POS changes, mean: {:.4f}%, median: {:.4f}%, '
              'min: {:.4f}%, max: {:.4f}%, std: {:.4f}%'
              .format(np.mean(change_distr), np.median(change_distr),
                      change_distr.min(), change_distr.max(),
                      change_distr.std()))

        if backend == 'nltk':
            sem_distr = np.array(wordnet_path_dist[category])
            print('\tSemantic path similarity of changes ({} total), '
                  'mean: {:.4f}, median: {:.4f}, '
                  'min: {:.4f}, max: {:.4f}, std: {:.4f}'
                  .format(len(sem_distr),
                          np.mean(sem_distr), np.median(sem_distr),
                          sem_distr.min(), sem_distr.max(), sem_distr.std()))
            sem_distr = np.array(wordnet_wup_dist[category])
            print('\tSemantic Wu-Palmer similarity of changes ({} total), '
                  'mean: {:.4f}, median: {:.4f}, '
                  'min: {:.4f}, max: {:.4f}, std: {:.4f}'
                  .format(len(sem_distr),
                          np.mean(sem_distr), np.median(sem_distr),
                          sem_distr.min(), sem_distr.max(), sem_distr.std()))

    print('\tMod vs least POS t-test:')
    print(ss.ttest_ind(np.array(n_pos_change['mod']),
                       np.array(n_pos_change['least'])))
    if backend == 'nltk':
        print('\tMod vs least path-similarity t-test:')
        print(ss.ttest_ind(np.array(wordnet_path_dist['mod']),
                           np.array(wordnet_path_dist['least'])))
        print('\tMod vs least WuP-similarity t-test:')
        print(ss.ttest_ind(np.array(wordnet_wup_dist['mod']),
                           np.array(wordnet_wup_dist['least'])))

def train_topic_model(seqs, vocabulary, n_components=10):
    seqs = np.array([ ' '.join(seq) for seq in sorted(seqs.keys()) ])

    X = dok_matrix((len(seqs), len(vocabulary)))
    for seq_idx, seq in enumerate(seqs):
        for word in seq.split(' '):
            X[seq_idx, vocabulary[word] - 1] += 1
    X = csr_matrix(X)

    from sklearn.decomposition import LatentDirichletAllocation as LDA
    tprint('LDA, {} components...'.format(n_components))
    model = LDA(n_components=n_components, n_jobs=10).fit(X)

    return model

def lda_topic_model(infos, n_components=10):
    from headlines import setup
    seqs, vocabulary = setup()

    topic_model = train_topic_model(seqs, vocabulary,
                                    n_components=n_components)

    X_orig = dok_matrix((len(infos), len(vocabulary)))
    for info_idx, info in enumerate(infos):
        for word in info['orig'].split(' '):
            X_orig[info_idx, vocabulary[word]] += 1
    X_orig_topics = topic_model.transform(X_orig)
    topic_orig = np.argmax(X_orig_topics, axis=1)

    categories = [ 'mod', 'least', 'most' ]
    for category in categories:
        tprint('Category: {}'.format(category))

        X_category = dok_matrix((len(infos), len(vocabulary)))
        for info_idx, info in enumerate(infos):
            for word in info[category].split(' '):
                X_category[info_idx, vocabulary[word]] += 1
        X_category_topics = topic_model.transform(X_category)
        topic_category = np.argmax(X_category_topics, axis=1)

        tprint('Changed topics: {} / {}'.format(
            sum(topic_orig != topic_category), len(topic_orig)
        ))


if __name__ == '__main__':
    log_fname = sys.argv[1]

    infos = []
    with open(log_fname) as f:
        for line in f:
            content = line.split(' | ')[-1].strip()
            if content.startswith(info_prefix):
                info = read_headline_info(f, content)
                infos.append(info)

    part_of_speech(infos, backend='nltk')
    part_of_speech(infos, backend='flair')
