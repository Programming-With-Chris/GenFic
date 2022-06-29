import os
import shutil


class FileWork:

    STORY_LIMIT = 50
    START_POINT = 50

    def read_by_author(self, author_name):
        text_array = []
        for filename in os.listdir("../fiction"):
            if filename.find(author_name) >= 0:
                text = (open(os.path.join("../fiction", filename),  encoding="utf8").read())
                text_array.append(text)
                continue
            else:
                continue
        return text_array

    def read_all(self):
        text_array = []
        print('reading files')
        i = 0
        for filename in os.listdir("../fiction/"):
            try:
                if i < self.STORY_LIMIT:
                    text = (open(os.path.join("../fiction/", filename),  encoding="utf8").read())
                    i = i + 1
                    text_array.append(text)
            except:
                continue
        return text_array


if __name__ == "__main__":
    #For testing filework()
    test = FileWork()
    return_array = test.read_by_author("Edgar Allan Poe")
    print(return_array)