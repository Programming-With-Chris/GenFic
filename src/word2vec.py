import gensim
from file import FileWork


class Word2VecWorker:


    processed_stories = []
    model = None

    ## text is the array of files
    '''def __init__(self, text, optionalAuth="word2vec"):
        path = os.path.join("../res/models/","")
        exists = os.path.isfile(path + optionalAuth + '.bin')
        if exists:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path + optionalAuth + '.bin', binary=True)
            print('model loaded')
        else:
            print('training Word2Vec now...')
            for story in text:
                processed_story = gensim.utils.simple_preprocess(story)
                self.processed_stories.append(processed_story)
            self.model = gensim.models.Word2Vec(
                self.processed_stories,
                size=100,
                window=10,
                min_count=2,
                workers=10)
            self.model.train(self.processed_stories, total_examples=len(self.processed_stories), epochs=10)
            self.model.wv.save_word2vec_format(path + optionalAuth + '.bin', binary=True)
            print('training complete')'''

    def cleanData(self, textArray):
        returnArray = []
        for story in textArray:
            processed_story = gensim.utils.simple_preprocess(story)
            returnArray.append(processed_story)
        return returnArray


    def vectorize(self, storyArray):
        processed_stories = []
        for story in storyArray:
            processed_story = gensim.utils.simple_preprocess(story)
            print(processed_story)
            processed_stories.append(processed_story)
        model = gensim.models.Word2Vec(
            processed_stories,
            size=30,
            window=10,
            min_count=2,
            workers=10)
        print('starting word2vec training')
        model.train(processed_stories, total_examples=len(processed_stories), epochs=10)
        print(model.wv)
        model.wv.save_word2vec_format('src/models/word2vec.bin', binary=True)
        print('training complete')



if __name__ == "__main__":
    test = FileWork()
    returnArray = test.read_by_author("Edgar Allan Poe")
    w2vTest = Word2VecWorker()
    w2vTest.vectorize(returnArray)