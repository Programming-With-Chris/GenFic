from model import *
from clean import *
from file import *
import random
import time

import tensorflow as tf
import numpy as np

maxlen = 30
BATCH_SIZE = 20
num_epochs = 40

def buildModel(words, dropout=0.3):
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Embedding(len(words), maxlen, input_length=maxlen))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), input_shape=(maxlen, len(words))))
    model.add(tf.keras.layers.LSTM(100, input_shape=(maxlen, len(words))))
    #model.add(tf.keras.layers.LSTM(100))
    #model.add(tf.keras.layers.LSTM(100))
    if dropout > 0:
        model.add(tf.layers.Dropout(dropout))
    model.add(tf.layers.Dense(len(words), activation='relu'))
    model.add(tf.keras.layers.Activation('softmax'))
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



def train_model():
    test = FileWork()
    storyArray = test.readByAuthor("Charles Dickens")
    #storyArray = test.readAll()
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
    file_path = "./checkpoints/checkpoint-{epoch:03d}-CharlesDickens.bin"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    callbacks_list = [checkpoint, early_stopping]

    model.fit_generator(generator(sentences, next_words, BATCH_SIZE, words, word_indices),
                        steps_per_epoch=int(len(sentences) / num_epochs) + 1,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, maxlen, words, word_indices),
                        validation_steps=int(len(sentences_test) / num_epochs) + 1)
    model.save('./models/model-' + time.strftime("%Y%m%d-%H%M%S") + ".h5")

def load_model(checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())
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
    sentence = seed.split(" ")
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + seed)

    print(seed)
    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        print(" "+next_word, end="")
        if i % 15 == 0:
            print("\n")
    print("\n")



def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = seed.split(" ")
    return_valid_words = ""
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
            return_valid_words = return_valid_words + w + " "
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return return_valid_words

if __name__ == "__main__":
    train_model()
    '''model = load_model('./checkpoints/checkpoint-004.bin')
    test = FileWork()
    storyArray = test.readByAuthor("Charles Dickens")
    #storyArray = test.readAll()
    wordArray = storiesToWordArray(storyArray)
    word_indices, indices_word, ignored_words, vocabulary = vectorization(wordArray)

    seed = "Pope's Satires, which still deal with characters of men, followed"
    "immediately, some appearing in a folio in January, 1735.  That part of"
    "the epistle to Arbuthnot forming the Prologue, which gives a character of"
    "Addison, as Atticus, had been sketched more than twelve years before, and"
    "earlier sketches of some smaller critics were introduced; but the"
    "beginning and the end, the parts in which Pope spoke of himself and of"
    "his father and mother, and his friend Dr. Arbuthnot, were written in 1733"
    "and 1734.  Then follows an imitation of the first Epistle of the Second"
    "Book of the Satires of Horace, concerning which Pope told a friend, When"
    "I had a fever one winter in town that confined me to my room for five or"
    "six days, Lord Bolingbroke, who came to see me, happened to take up a"
    "Horace that lay on the table, and, turning it over, dropped on the first"
    "satire in the Second Book, which begins, 'Sunt, quibus in satira.'  He"
    "observed how well that would suit my case if I were to imitate it in"
    "English.  After he was gone, I read it over, translated it in a morning"
    "or two, and sent it to press in a week or a fortnight after (February,"
    "1733).  And this was the occasion of my imitating some others of the"
    "Satires and Epistles.  The two dialogues finally used as the Epilogue to"
    "the Satires were first published in the year 1738, with the name of the"
    "year, Seventeen Hundred and Thirty-eight.  Samuel Johnson's London,"
    "his first bid for recognition, appeared in the same week, and excited in"
    "Pope not admiration only, but some active endeavour to be useful to its"
    "author."

   #seed = ""
    #for i in range((len(vocabulary) - 50)):
    #    pos = random.randint(0,len(vocabulary) - 1)
    #    seed = seed + vocabulary[pos] + " "

    print(seed)
    #seed = seed[-200:]

    seed = validate_seed(vocabulary, seed)
    generate_text(model, indices_word, word_indices, seed, maxlen, 1.0, 250, vocabulary)'''
