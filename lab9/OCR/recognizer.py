from collections import defaultdict

from scipy import misc, ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


class Recognizer():
    def __init__(self, imagePath='data/sans/facebook.png', sans=True):
        self.image = self.readImage(imagePath)
        self.fontImages = self.getLettersImages(sans)
        self.lettersPositions = {}
        self.maxf = 7
        self.orderAndFreq = [('a', 1),
                             ('f', 1),
                             ('g', 1),
                             ('k', 1),
                             ('m', 1),
                             ('p', 1),
                             ('q', 1),
                             ('s', 1),
                             ('t', 1),
                             ('u', 1),
                             ('w', 1),
                             ('x', 1),
                             ('y', 1),
                             ('z', 1),
                             ('b', 2),
                             ('h', 2),
                             ('j', 2),
                             ('v', 2),
                             ('d', 3),
                             ('e', 3),
                             ('n', 3),
                             ('r', 3),
                             ('i', 4),
                             ('o', 4),
                             ('l', 5),
                             ('c', 7)]

    def readImage(self, path):
        result = ndimage.imread(path, mode='I')
        return result

    def invertImage(self, image):
        return 255 - image

    def showImage(self, image, size):
        plt.figure(figsize=(size, size))
        plt.imshow(image)
        plt.show()

    def getPathDict(self, sans):
        path = 'data/sans/' if sans else 'data/serif/'
        letters = [chr(i) for i in range(97, 123)]
        sansPathDict = {}
        for l in letters:
            sansPathDict[l] = path + l + '.png'
        return sansPathDict

    def getLettersImages(self, sans):
        result = {}
        paths = self.getPathDict(sans)
        letters = [chr(i) for i in range(97, 123)]
        images = [self.invertImage(self.readImage(v)) for k, v in paths.items()]
        for letter, i in zip(letters, range(len(letters))):
            result[letter] = images[i]
        return result

    def correlateLetter(self, img, letter, color, threshold=0.83):
        fi = np.fft.fft2(self.invertImage(img))
        fp = np.fft.fft2(np.rot90(letter, 2), fi.shape)
        m = np.multiply(fi, fp)
        corr = np.fft.ifft2(m)
        corr = np.abs(corr)
        corr = corr.astype(float)
        corr[corr < threshold * np.amax(corr)] = 0
        corr[corr != 0] = color
        return corr

    def getLetterPositions(self, correlation, letter):
        positions = []
        lw, lh = letter.shape
        lw += 2
        lh += 5
        tmpi, tmpj = (-1 * lw, -1 * lh)
        for (i, j), val in np.ndenumerate(correlation):
            if val > 0.0 and not (tmpi + lw > i and tmpj + lh > j):
                positions.append((i, j))
                tmpi, tmpj = i, j
        return positions

    def removeLetter(self, letter, positions):
        for x, y in positions:
            lw, lh = self.fontImages[letter].shape
            lh += 5
            lw += 2
            self.image[x - lw:x, y - lh:y] = 255


    def matchAllLetters(self):
        for element in self.orderAndFreq:
            start = time.clock()
            letter = element[0]
            letterImage = self.fontImages[letter]
            corrCoefficient = 10 * element[1] / self.maxf / 100
            correlation = self.correlateLetter(self.image, letterImage, 254, threshold=0.85 + corrCoefficient)
            self.lettersPositions[letter] = self.getLetterPositions(correlation, letterImage)
            self.removeLetter(letter, self.lettersPositions[letter])
            print(letter, " TIME : ", time.clock() - start)

    def positionsToText(self):
        positions = []
        for k, v in self.lettersPositions.items():
            for e in v:
                positions.append((k, e[0] - e[0]%100, e[1]))
        positions = sorted(positions, key=lambda e: (e[1], e[2]))
        print(positions)
        text = ""
        space = 70
        line = 99
        for i in range(len(positions)-1):
            text += positions[i][0]
            if abs(positions[i][2] - positions[i+1][2]) > space:
                text += chr(32)
            if abs(positions[i][1] - positions[i+1][1]) >= line:
                text += '\n'
        print(text)



if __name__ == '__main__':
    start = time.clock()
    oceery = Recognizer()
    oceery.matchAllLetters()
    oceery.positionsToText()
    print("TIME AFTER MATCH  nbh: ", time.clock() - start)

    # frq = [(k, len(v)) for k, v in oceery.lettersPositions.items()]
    # print(frq)
    # dic = np.load('data/alphabetMatch.npy')
    # dic = sorted(dic, key=lambda e: e[1])
    # res = []
    # for i in range(len(dic)):
    #     res.append((dic[i][0], int(dic[i][1])))
    # print(res)

    print(time.clock() - start)
