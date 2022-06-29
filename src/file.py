import os
import shutil


class FileWork:

    STORY_LIMIT = 50
    START_POINT = 50

    def readByAuthor(self, authorName):
        textArray = []
        #for filename in os.listdir("../res/Gutenberg/txt/"):
        for filename in os.listdir("C:\\Users\\CC163\\PycharmProjects\\GenFic\\fiction"):
            if filename.find(authorName) >= 0:
                text = (open(os.path.join("C:\\Users\\CC163\\PycharmProjects\\GenFic\\fiction", filename),  encoding="utf8").read())
                textArray.append(text)
                continue
            else:
                continue
        return textArray

    def readAll(self):
        textArray = []
        print('reading files')
        i = 0
        for filename in os.listdir("../fiction/"):
            try:
                if i < self.STORY_LIMIT:
                    text = (open(os.path.join("../fiction/", filename),  encoding="utf8").read())
                    i = i + 1
                    textArray.append(text)
            except:
                continue
        return textArray


if __name__ == "__main__":
    test = FileWork()
    returnArray = test.readByAuthor("Abraham Lincoln")
    print(returnArray)