import tensorflow as tf


def word2vec():
# Phase1: assemble your graph
    # Step 1: define the placeholders for input and output
    with tf.name_scope("data"):
        
    # Step 2: define weights
    with tf.name_scope("weight"):
        
    # Step 3: define inference model        
    with tf.name_scope("loss"):
        
        # Step 4: define the loss function
    
    # Step 5: define the optimizer
    
# Phase2: ececute the computation
    with tf.Session() as sess:

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()        