import time
import argparse
import collections

from gensim.models import word2vec

import utils.my_io as my_io


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", type=str, default='/Users/wind/WORK/public_data_set/text8/text8')
    parse.add_argument("--epoches", type=int, default=35000)
    parse.add_argument("--vocab_size", type=int, default=10000)
    parse.add_argument("--embedding_size", type=int, default=300)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--num_skips", type=int, default=1)
    parse.add_argument("--skip_window", type=int, default=2)
    parse.add_argument("--valid_size", type=int, default=16)
    parse.add_argument("--valid_window", type=int, default=100)
    parse.add_argument("--num_sampled", type=int, default=20)
    parse.add_argument("--nce", type=bool, default=True)

    args = parse.parse_args()
    print args

    data_raw = my_io.read_text_file(args.dataset)
    # dataset, count, dictionary, reversed_dictionary = build_dataset(data_raw[0], args.vocab_size)
    t_s = time.time()
    model = word2vec.Word2Vec(data_raw[0], iter=10, min_count=10, size=300, workers=4)
    print('Time collapse: %d', time.time() - t_s)