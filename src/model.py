from model import *
from clean import *
from config import *

import tensorflow as tf
import numpy as np

maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()


def build_model(words):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(words)), activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(words)), activation='relu'))

    model.add(tf.keras.layers.Dense(len(words), activation='softmax'))
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_model(story_array, model_save_location):

    next_words, next_words_test, sentences, sentences_test, word_indices, words = build_word_arrays(story_array)

    model = build_model(words)
    print('compiling model')
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    file_path = "./checkpoints/checkpoint-{epoch:03d}-EdgarAllanPoe.bin"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)
    callbacks_list = [early_stopping]

    model.fit(vectorization(sentences, next_words, BATCH_SIZE, words, word_indices),
              steps_per_epoch=int(len(sentences) / num_epochs) + 1,
              epochs=num_epochs,
              callbacks=callbacks_list,
              validation_data=vectorization(sentences_test, next_words_test, maxlen, words, word_indices),
              validation_steps=int(len(sentences_test) / num_epochs) + 1)
    model.save(model_save_location)
    return model


def build_word_arrays(story_array):
    print('done reading')
    word_array = stories_to_word_array(story_array)
    print('done with word array')
    word_indices, indices_word, ignored_words, words = build_word_dictionaries(word_array)
    print('done vectorizing')
    sentences, next_words = create_sequences(word_array, ignored_words)
    print('done with sequences')
    sentences_train, next_words_train, sentences_test, next_words_test = shuffle_and_split_training_set(sentences,
                                                                                                        next_words)
    return next_words, next_words_test, sentences, sentences_test, word_indices, words


def load_model(checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())
    return model


def load_model_and_continue(story_array, checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())

    train_model(model, story_array, checkpoint_location)
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

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

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


def filter_valid_words(vocabulary, seed):
    return_valid_words = ""
    for w in seed:
        if w in vocabulary:
            return_valid_words = return_valid_words + w + " "
    return return_valid_words
