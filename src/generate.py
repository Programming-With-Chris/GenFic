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

    text_files = FileWork()
    story_array = text_files.read_by_author("Edgar Allan Poe")

    #train_model(story_array, 'C:\\Users\\CC163\\PycharmProjects\\GenFic\\models\\model-50-10-v3.h5')
    model = load_model('C:\\Users\\CC163\\PycharmProjects\\GenFic\\models\\model-50-10-v3.h5')

    word_array = stories_to_word_array(story_array)
    word_indices, indices_word, ignored_words, vocabulary = build_word_dictionaries(word_array)

    seed_starting_pos = rand.randint(0, len(word_array))
    seed = word_array[seed_starting_pos: seed_starting_pos + 400]

    seed = filter_valid_words(vocabulary, seed)
    generate_text(model, indices_word, word_indices, seed, maxlen, 1.0, 250, vocabulary)
