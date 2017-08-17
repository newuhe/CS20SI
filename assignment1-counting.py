# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 07:24:07 2017

@author: Administrator
"""
import tensorflow as tf
import numpy as np
import os
import zipfile
import random
from collections import Counter

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
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary
    
def build_co_matrix(words,vocab_size,dictionary):
    """ Build co-occurrence matrix """
    matrix=np.zeros()
    for word in words:
        
        
def svd():
    with tf.name_scope("data"):
        matrix=tf.placeholder()

        embed_matrix=tf.svd(matrix)
    
    with tf.Session() as sess:
        co_matrix=build_co_matrix()
        embed_matri=sess.run([embed_matrix],feed_dict={matrix:co_matrix})
    
def main():
    svd()

if __name__ == '__main__':
    main()        