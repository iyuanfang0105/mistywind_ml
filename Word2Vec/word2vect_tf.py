import time
import collections
import random
import math
import argparse

import numpy as np
import tensorflow as tf

import utils.my_utils as my_io

data_index = 0


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


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


def tf_word2vec_model(args):
    train_targets = tf.placeholder(tf.int32, shape=[args.batch_size], name='train_targets')
    train_contexts = tf.placeholder(tf.int32, shape=[args.batch_size, 1], name='train_contexts')

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    # valid_size = 16  # Random set of words to evaluate similarity on.
    # valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_dataset = tf.placeholder(tf.int32, shape=[args.valid_size], name='valid_samples')

    embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.embedding_size], -1.0, 1.0), name='embeddings')
    embed = tf.nn.embedding_lookup(embeddings, train_targets)

    if args.nce:
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([args.vocab_size, args.embedding_size], stddev=1.0 / math.sqrt(args.embedding_size)),
            name='nce_wights')
        nce_biases = tf.Variable(tf.zeros([args.vocab_size]), name='nce_bias')

        nce_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_contexts, inputs=embed,
                           num_sampled=args.num_sampled, num_classes=args.vocab_size))
        loss = nce_loss
    else:
        weights = tf.Variable(
            tf.truncated_normal([args.vocab_size, args.embedding_size], stddev=1.0 / math.sqrt(args.embedding_size)))
        biases = tf.Variable(tf.zeros([args.vocab_size]))
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

        train_contexts_one_hot = tf.one_hot(train_contexts, args.vocab_size)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_contexts_one_hot))
        loss = cross_entropy

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    return train_targets, train_contexts, valid_dataset, loss, optimizer, similarity, normalized_embeddings


def train(data, dict, args):
    with tf.Session() as sess:
        train_targets, train_contexts, valid_dataset, loss, optimizer, similarity, normalized_embeddings = tf_word2vec_model(
            args)
        tf.global_variables_initializer().run()

        loss_record = []
        average_loss = 0
        for epoch in range(args.epoches):
            target, context = generate_batch(data, args.batch_size, args.num_skips, args.skip_window)
            feed_dict = {train_targets: target, train_contexts: context}

            _, loss_v = sess.run([optimizer, loss], feed_dict=feed_dict)
            loss_record.append(loss_v)
            average_loss += loss_v

            if epoch % 2000 == 0:
                if epoch > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', epoch, ': ', average_loss)
                average_loss = 0
            if epoch % 10000 == 0:
                valid_examples = np.random.choice(args.valid_window, args.valid_size, replace=False)
                sim = sess.run(similarity, feed_dict={valid_dataset: valid_examples})
                for i in range(args.valid_size):
                    valid_word = dict[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = dict[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

        return normalized_embeddings.eval(), loss_record


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
    dataset, count, dictionary, reversed_dictionary = build_dataset(data_raw[0], args.vocab_size)
    t_s = time.time()
    embeddings, loss_record = train(dataset, reversed_dictionary, args)
    print('Time collapse: %d', time.time() - t_s)
