from utils import *

def semantic_headlines(dirname):
    datasets = [ '1', 'A', '2', 'B' ]
    responses = [ '1', '2' ]

    data = []
    for dataset in datasets:
        fname = ('{}/semantic_headlines_labeled_{}.csv'
                 .format(dirname, dataset))
        headline_type = {}
        with open(fname) as f:
            for line in f:
                fields = line.rstrip().split(',')
                headline_type[fields[1]] = fields[0]

        response_gram = { response: {} for response in responses }
        for response in responses:
            fname = ('{}/semantic_headlines_{}_response{}.csv'
                     .format(dirname, dataset, response))
            with open(fname) as f:
                f.readline() # Consume header.
                for line in f:
                    validity, headline = line.rstrip().split(',')
                    response_gram[response][headline] = (validity.lower() == 'x')

        for headline in headline_type.keys():
            data.append([
                dataset, headline_type[headline],
                response_gram['1'][headline], response_gram['2'][headline],
            ])

    df = pd.DataFrame(data, columns=[
        'dataset', 'type', 'response1', 'response2'
    ])

    n_agree = sum([
        r1 == r2 for r1, r2 in zip(df.response1, df.response2)
    ])
    human_agreement = n_agree / float(len(df))
    print('Human agreement: {} / {} = {:.2f}%\n'
          .format(n_agree, len(df), human_agreement * 100))

    types = sorted(set(df.type))
    for typ in types:
        print('Type: {}'.format(typ))
        n_gram_1 = sum(df[df.type == typ].response1)
        n_gram_2 = sum(df[df.type == typ].response2)
        n_gram_12 = sum([
            r1 == r2 == True for r1, r2 in
            zip(df[df.type == typ].response1,
                df[df.type == typ].response2)
        ])
        n_agree_type = sum([
            r1 == r2 for r1, r2 in
            zip(df[df.type == typ].response1,
                df[df.type == typ].response2)
        ])
        print('Grammatical (Response 1): {}'.format(n_gram_1))
        print('Grammatical (Response 2): {}'.format(n_gram_2))
        print('Grammatical (Response 1 and 2): {}'.format(n_gram_12))
        print('Grammatical agreement: {}'.format(n_agree_type))
        print('')

if __name__ == '__main__':
    semantic_headlines('data/headlines/semantic_headlines')
