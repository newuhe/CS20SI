import tensorflow as tf
import numpy as np
import os
import zipfile
import random
from collections import Counter
from tensorflow.contrib.tensorboard.plugins import projector


VOCAB_SIZE = 10000
BATCH_SIZE = 128
EMBED_SIZE = 300
DISPLAY_STEP=2000
TRAIN_STEPS=10000
NUM_SAMPLED = 64    # Number of negative examples to sample.
SKIP_WINDOW=3
LEARNING_RATE = 1.0

#process data
def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocabulary(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    os.makedirs('processed')
    with open('./processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch
        
def process_data(vocab_size, batch_size, skip_window):
    words = read_data("./text8.zip")
    dictionary, _ = build_vocabulary(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)

#build a model
def word2vec(batch_gen):
# Phase1: assemble your graph
    # Step 1: define the placeholders for input and output
    with tf.name_scope("data"):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')
    # Step 2: define weights
    with tf.name_scope("weight"):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), 
                            name='embed_matrix')
    # Step 3: define inference model        
    with tf.name_scope("loss"):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)), 
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
        # Step 4: define the loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=target_words, 
                                            inputs=embed, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
# Phase2: ececute the computation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs/', sess.graph)
        total_loss=0
        
        for index in range(TRAIN_STEPS):
            centers,targets=next(batch_gen)
            batch_loss, _ = sess.run([loss, optimizer], 
                                    feed_dict={center_words: centers, target_words: targets})
            total_loss += batch_loss
            if (index + 1) % DISPLAY_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / DISPLAY_STEP))
                total_loss = 0.0
                
        writer.close()
        
# Phase3: visulize       
        # code to visualize the embeddings. uncomment the below to visualize embeddings
        # run "'tensorboard --logdir='processed'" to see the embeddings
        final_embed_matrix = sess.run(embed_matrix) #result of sess.run is constant rather than variable
        
        # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        
        # # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'processed/vocab_1000.tsv'

        # # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()        