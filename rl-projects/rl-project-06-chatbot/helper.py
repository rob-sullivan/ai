#seq2seq dialog generator used for the reverse model of backward entropy loss.
# This determines the reward for semantic coherence in the policy gradient dialogue.
# In other words it helps represent future reward based on LSTM (i.e encoding, decoding and generating builds).
# feature extraction script gets features and characteristics from data to help improve training.

# libraries
import numpy as np
import re
#needed for tensorflow to access gpu
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
import tensorflow as tf

# if RL is set to true a scaler is computed based on semantic coherence and ease of answering loss caption.

def model_inputs(embed_dim, reinforcement= False):    
    word_vectors = tf.placeholder(tf.float32, [None, None, embed_dim], name = "word_vectors")
    reward = tf.placeholder(tf.float32, shape = (), name = "rewards")
    caption = tf.placeholder(tf.int32, [None, None], name = "captions")
    caption_mask = tf.placeholder(tf.float32, [None, None], name = "caption_masks")
    if reinforcement: # Normal training returns only the word_vectors, caption and caption_mask placeholders, 
        # With reinforcement learning, there is an extra placeholder for rewards
        return word_vectors, caption, caption_mask, reward
    else:
        return word_vectors, caption, caption_mask

# performs encoding for sequence to sequence network. The input sequence is passed into the encoder and returns both an ouput RNN and the state
def encoding_layer(word_vectors, lstm_size, num_layers, keep_prob, vocab_size):
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size), keep_prob) for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(cells, 
                                        word_vectors, 
                                       dtype=tf.float32)
    return outputs, state

# training process for decoder using LSTM cells and encoder states 
def decode_train(enc_state, dec_cell, dec_input, 
                         target_sequence_length,output_sequence_length,
                         output_layer, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,                #Apply dropout to the LSTM cell
                                             output_keep_prob=keep_prob)
    
    
    helper = tf.contrib.seq2seq.TrainingHelper(dec_input,             #Training helper for decoder 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              enc_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True,
                                                     maximum_iterations=output_sequence_length)
    return outputs

# generates an inference decoder that makes use of greedy helpter which feeds last output of decoder as next decoder input.
# returns training logits and sample id
def decode_generate(encoder_state, dec_cell, dec_embeddings,
                         target_sequence_length,output_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], 1),  #Decoder helper for inference
                                                      2)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True,
                                                     maximum_iterations=output_sequence_length)
    return outputs

# decodes the encoded layer
def decoding_layer(dec_input, enc_state,
                   target_sequence_length,output_sequence_length,
                   lstm_size,
                   num_layers,n_words,
                   batch_size, keep_prob,embedding_size, Train = True):
    target_vocab_size = n_words
    with tf.device("/cpu:0"):
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,embedding_size], -0.1, 0.1), name='Wemb')
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
    
    if Train:
        with tf.variable_scope("decode"):
            train_output = decode_train(enc_state, 
                                                cells, 
                                                dec_embed_input, 
                                                target_sequence_length, output_sequence_length,
                                                output_layer, 
                                                keep_prob)

    with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        infer_output = decode_generate(enc_state, 
                                            cells, 
                                            dec_embeddings, target_sequence_length,
                                           output_sequence_length,
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)
    if Train:
        return train_output, infer_output
    return infer_output

# appends the index corresponding to <bos> which is the beginning of a sentence to the first index of the capton tensor for every batch.
def bos_inclusion(caption,batch_size):
 
    sliced_target = tf.strided_slice(caption, [0,0], [batch_size, -1], [1,1])
    concat = tf.concat([tf.fill([batch_size, 1],1), sliced_target],1)
    return concat

# creates an array of size maxlen from every question by padding with zeros or trucating where applicable
def pad_sequences(questions, sequence_length =22):
    lengths = [len(x) for x in questions]
    num_samples = len(questions)
    x = np.zeros((num_samples, sequence_length)).astype(int)
    for idx, sequence in enumerate(questions):
        if not len(sequence):
            continue  # empty list/array was found
        truncated  = sequence[-sequence_length:]

        truncated = np.asarray(truncated, dtype=int)

        x[idx, :len(truncated)] = truncated

# only takes non numerical data
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    data = ' '.join(words)
    return data

# create batches to feed into network from word vector representation
def make_batch_input(batch_input, input_sequence_length, embed_dims, word2vec):
    
    for i in range(len(batch_input)):
        
        batch_input[i] = [word2vec[w] if w in word2vec else np.zeros(embed_dims) for w in batch_input[i]]
        if len(batch_input[i]) >input_sequence_length:
            batch_input[i] = batch_input[i][:input_sequence_length]
        else:
            for _ in range(input_sequence_length - len(batch_input[i])):
                batch_input[i].append(np.zeros(embed_dims))
    return np.array(batch_input)

def replace(target,symbols):  #Remove symbols from sequence
    for symbol in symbols:
        target = list(map(lambda x: x.replace(symbol,''),target))
    return target
                      
def make_batch_target(batch_target, word_to_index, target_sequence_length):
    target = batch_target
    target = list(map(lambda x: '<bos> ' + x, target))
    symbols = ['.', ',', '"', '\n','?','!','\\','/']
    target = replace(target, symbols)

    for idx, each_cap in enumerate(target):
        word = each_cap.lower().split(' ')
        if len(word) < target_sequence_length:
            target[idx] = target[idx] + ' <eos>'  #Append the end of symbol symbol 
        else:
            new_word = ''
            for i in range(target_sequence_length-1):
                new_word = new_word + word[i] + ' '
            target[idx] = new_word + '<eos>'
            
    target_index = [[word_to_index[word] if word in word_to_index else word_to_index['<unk>'] for word in 
                          sequence.lower().split(' ')] for sequence in target]
    #print(target_index[0])
    
    caption_matrix = pad_sequences(target_index,target_sequence_length)
    caption_matrix = np.hstack([caption_matrix, np.zeros([len(caption_matrix), 1])]).astype(int)
    caption_masks = np.zeros((caption_matrix.shape[0], caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum(), caption_matrix)))
    #print(nonzeros)
    #print(caption_matrix[1])
    
    for ind, row in enumerate(caption_masks): #Set the masks as an array of ones where actual words exist and zeros otherwise
        row[:nonzeros[ind]] = 1                 
        #print(row)
    print(caption_masks[0])
    print(caption_matrix[0])
    return caption_matrix,caption_masks   

def generic_batch(generic_responses, batch_size, word_to_index, target_sequence_length):
    size = len(generic_responses) 
    if size > batch_size:
        generic_responses = generic_responses[:batch_size]
    else:
        for j in range(batch_size - size):
            generic_responses.append('')
    return make_batch_target(generic_responses, word_to_index, target_sequence_length)

# generate sentences from the predicted indices of word with the next highest probability
def index2sentence(generated_word_index, prob_logit, ixtoword):
    generated_word_index = list(generated_word_index)
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3 or generated_word_index[i] == 0:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1

            generated_word_index[i] = curindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])    
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')  #Replace the beginning of sentence tag
    generated_sentence = generated_sentence.replace('<eos>', '')   #Replace the end of sentence tag
    generated_sentence = generated_sentence.replace('--', '')      #Replace the other symbols predicted
    generated_sentence = generated_sentence.split('  ')
    for i in range(len(generated_sentence)):       #Begin sentences with Upper case 
        generated_sentence[i] = generated_sentence[i].strip()
        if len(generated_sentence[i]) > 1:
            generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
        else:
            generated_sentence[i] = generated_sentence[i].upper()
    generated_sentence = ' '.join(generated_sentence)
    generated_sentence = generated_sentence.replace(' i ', ' I ')
    generated_sentence = generated_sentence.replace("i'm", "I'm")
    generated_sentence = generated_sentence.replace("i'd", "I'd")

    return generated_sentence