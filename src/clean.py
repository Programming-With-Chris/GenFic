import numpy as np
from config import *


maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()

def storiesToWordArray(storyArray):
    wordArray = []
    for story in storyArray:
        story = story.replace('.', ' . ')
        #story = story.replace('(', ' ( ')
        #story = story.replace(')', ' ) ')
        story = story.replace(',', ' , ')
        story = story.replace('!', ' ! ')
        story = story.replace('"', ' " ')
        story = story.replace('\'', ' \' ')
        splitStory = story.split()
        for word in splitStory:
            wordArray.append(word)
    return wordArray


def combineStories(storyArray):
    outString = ""
    for story in storyArray:
        outString = outString + story
    return outString

def vectorization(wordArray):
    word_freq = {}
    for word in wordArray:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)
    words = set(wordArray)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))
    return word_indices, indices_word, ignored_words, words


def shuffle_and_split_training_set(sentences_original, labels_original, percentage_test=3):
    # shuffle at unison
    print('Shuffling sentences')
    tmp_sentences = []
    tmp_next_char = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_char.append(labels_original[i])
    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_char[:cut_index], tmp_next_char[cut_index:]

    print("Training set = %d\nTest set = %d" % (len(x_train), len(y_test)))
    return x_train, y_train, x_test, y_test

def createSequences(wordArray, ignored_words):
    step = 1
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(wordArray) - maxlen, step):
        if len(set(wordArray[i: i+maxlen+1]).intersection(ignored_words)) == 0:
            sentences.append(wordArray[i: i + maxlen])
            next_words.append(wordArray[i + maxlen])
        else:
            ignored = ignored + 1

    print('Ignored sentences: ',ignored)
    print('Remaining sequences:', len(sentences))
    return sentences, next_words