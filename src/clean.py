import numpy as np
from config import *

maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()


def stories_to_word_array(story_array):
    word_array = []
    for story in story_array:
        story = story.replace('.', ' . ')
        # story = story.replace('(', ' ( ')
        # story = story.replace(')', ' ) ')
        story = story.replace(',', ' , ')
        story = story.replace('!', ' ! ')
        story = story.replace('"', ' " ')
        story = story.replace('\'', ' \' ')
        split_story = story.split()
        for word in split_story:
            word_array.append(word)
    return word_array


def combine_stories(story_array):
    out_string = ""
    for story in story_array:
        out_string = out_string + story
    return out_string


def build_word_dictionaries(word_array):
    word_freq = {}
    for word in word_array:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)
    words = set(word_array)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))
    return word_indices, indices_word, ignored_words, words


def shuffle_and_split_training_set(sentences_original, labels_original, percentage_test=3):
    tmp_sentences = []
    tmp_next_char = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_char.append(labels_original[i])
    cut_index = int(len(sentences_original) * (1. - (percentage_test / 100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_char[:cut_index], tmp_next_char[cut_index:]

    print("Training set = %d\nTest set = %d" % (len(x_train), len(y_test)))
    return x_train, y_train, x_test, y_test


# TODO add step to config, also remember that you changed it from 1 to 3
def create_sequences(word_array, ignored_words):
    step = 3
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(word_array) - maxlen, step):
        if len(set(word_array[i: i + maxlen + 1]).intersection(ignored_words)) == 0:
            sentences.append(word_array[i: i + maxlen])
            next_words.append(word_array[i + maxlen])
        else:
            ignored = ignored + 1
    print('Ignored sentences: ', ignored)
    print('Remaining sequences:', len(sentences))
    return sentences, next_words


def vectorization(sentence_list, next_word_list, batch_size, word_array, word_indices):
    index = 0
    while True:
        x = np.zeros((batch_size, maxlen, len(word_array)), dtype=np.bool)
        y = np.zeros((batch_size, len(word_array)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1

            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y