from model import *
from clean import *
from file import *
from config import *

import tensorflow as tf
import numpy as np

maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()

def buildModel(words, dropout=0.3):
    model = tf.keras.Sequential()
    #model.add(
    #    tf.keras.layers.Embedding(batch_input_shape=(len(words), maxlen), input_dim=len(words), output_dim=len(words),
    #                             input_length=maxlen))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), input_shape=(maxlen, len(words))))
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(words)), activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(words)), activation='relu'))
    #if dropout > 0:
    #    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(len(words), activation='softmax'))
    #model.add(tf.keras.layers.Activation('softmax'))
    return model


def generator(sentence_list, next_word_list, batch_size, wordArray, word_indices):
    index = 0
    while True:
        x = np.zeros((batch_size, maxlen, len(wordArray)), dtype=np.bool)
        y = np.zeros((batch_size, len(wordArray)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1

            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_model(model_save_location):
    text_files = FileWork()
    storyArray = text_files.readByAuthor("Edgar Allan Poe")
    # storyArray = test.readAll()
    print('done reading')
    wordArray = storiesToWordArray(storyArray)
    print('done with word array')
    word_indices, indices_word, ignored_words, words = vectorization(wordArray)
    print('done vectorizing')
    sentences, next_words = createSequences(wordArray, ignored_words)
    print('done with sequences')
    sentences_train, next_words_train, sentences_test, next_words_test = shuffle_and_split_training_set(sentences,
                                                                                                        next_words)
    print('building model')
    model = buildModel(words)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    # file_path = "./checkpoints/checkpoint-{epoch:03d}-EdgarAllanPoe.bin"
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)
    callbacks_list = [early_stopping]

    model.fit(generator(sentences, next_words, BATCH_SIZE, words, word_indices),
                        steps_per_epoch=int(len(sentences) / num_epochs) + 1,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, maxlen, words, word_indices),
                        validation_steps=int(len(sentences_test) / num_epochs) + 1)
    model.save(model_save_location)


def load_model(checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())
    return model


def load_model_and_continue(checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())
    test = FileWork()
    storyArray = test.readByAuthor("Edgar Allan Poe")
    # storyArray = test.readAll()
    print('done reading')
    wordArray = storiesToWordArray(storyArray)
    print('done with word array')
    word_indices, indices_word, ignored_words, words = vectorization(wordArray)
    print('done vectorizing')
    sentences, next_words = createSequences(wordArray, ignored_words)
    print('done with sequences')
    sentences_train, next_words_train, sentences_test, next_words_test = shuffle_and_split_training_set(sentences,
                                                                                                        next_words)
    print('building model')
    # model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    # print(model.summary())
    file_path = "./checkpoints/checkpoint-{epoch:03d}-EdgarAllanPoe.bin"

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    # callbacks_list = [checkpoint, early_stopping]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=50)
    callbacks_list = [early_stopping]

    model.fit_generator(generator(sentences, next_words, BATCH_SIZE, words, word_indices),
                        steps_per_epoch=int(len(sentences) / num_epochs) + 1,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, maxlen, words, word_indices),
                        validation_steps=int(len(sentences_test) / num_epochs) + 1)
    model.save(checkpoint_location)
    return model


def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity, vocabulary):
    """
    Similar to lstm_train::on_epoch_end
    Used to generate text using a trained model
    :param model: the trained Keras model (with model.load)
    :param indices_word: a dictionary pointing to the words
    :param seed: a string to be used as seed (already validated and padded)
    :param sequence_length: how many words are given to the model to generate
    :param diversity: is the "temperature" of the sample function (usually between 0.1 and 2)
    :param quantity: quantity of words to generate
    :return: Nothing, for now only writes the text to console
    """
    sentence = seed.split()
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + seed)


    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            if t < sequence_length:
                x_pred[0, t, word_indices[word]] = 1

        #TODO Probably the place to change
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        # if next_word is ' , ' or ' . ':
        #   next_word = next_word[-1:]

        sentence = sentence[1:]
        sentence.append(next_word)

        punctuation_array = [".", ",", "!", "?"]

        if next_word in punctuation_array:
            print(next_word, end="")
        else:
            print(" " + next_word, end="")

        if i % 15 == 0:
            print("\n")
    print("\n")


def validate_seed(vocabulary, seed):
    #print("\nValidating that all the words in the seed are part of the vocabulary: ")
    return_valid_words = ""
    for w in seed:
        #print(w, end="")
        if w in vocabulary:
            #print(" ✓ in vocabulary")
            return_valid_words = return_valid_words + w + " "
        else:
            #print(" ✗ NOT in vocabulary")
            valid = False
    return return_valid_words



