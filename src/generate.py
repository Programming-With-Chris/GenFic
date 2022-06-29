from config import *
from model import *
from clean import *
from file import *
import time

import random as rand
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #train_model('C:\\Users\\CC163\\PycharmProjects\\GenFic\\models\\model-50-10-test.h5')
    #load_model_and_continue('C:\\Users\\CC163\\PycharmProjects\\GenFic\\models\\model-EAP-test.h5')
    # model = load_model('./models/model-20190416-194355.h5')
    model = load_model('C:\\Users\\CC163\\PycharmProjects\\GenFic\\models\\model-50-10-v3.h5')
    test = FileWork()
    storyArray = test.readByAuthor("Edgar Allan Poe")

    wordArray = storiesToWordArray(storyArray)
    word_indices, indices_word, ignored_words, vocabulary = vectorization(wordArray)

    seed_starting_pos = rand.randint(0, len(wordArray))
    seed = wordArray[seed_starting_pos: seed_starting_pos+300]

    seed = validate_seed(vocabulary, seed)
    generate_text(model, indices_word, word_indices, seed, maxlen, 1.0, 250, vocabulary)