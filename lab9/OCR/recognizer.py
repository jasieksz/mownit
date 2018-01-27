from collections import defaultdict
from functools import reduce
from scipy import misc, ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


class Recognizer():
    def __init__(self, imagePath='data/sans/facebook2.png', sans=True):
        self.image = self.readImage(imagePath)
        self.fontImages = self.getLettersImages(sans)
        self.lettersPositions = {}
        self.maxf = 8
        self.orderAndFreq = [('x', 1), ('z', 1), ('w', 1), ('y', 1), ('f', 1), ('k', 1), ('g', 8),
                             ('b', 2), ('p', 8), ('m', 1), ('a', 1), ('t', 1), ('u', 1), ('s', 1),
                             ('j', 2), ('v', 2), ('h', 2), ('d', 8), ('l', 8), ('e', 8), ('c', 18),
                             ('n', 3), ('r', 3), ('i', 4), ('o', 8), ('q', 8)]

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
        for i in range(10):
            sansPathDict[str(i)] = path + str(i) + '.png'

        return sansPathDict

    def getLettersImages(self, sans):
        result = {}
        paths = self.getPathDict(sans)
        letters = [chr(i) for i in range(97, 123)]
        for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            letters.append(i)
        images = [self.invertImage(self.readImage(v)) for k, v in paths.items()]
        for letter, i in zip(letters, range(len(letters))):
            result[letter] = images[i]
        return result

    def correlateLetter(self, img, letter, color, threshold=0.85):
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
        tmpi, tmpj = (-1 * lw, -1 * lh)
        for (i, j), val in np.ndenumerate(correlation):
            if val > 0.0 and not (tmpi + lw > i and tmpj + lh > j):
                positions.append((i, j))
                tmpi, tmpj = i, j
        return positions

    def removeLetter(self, letter, positions):
        for x, y in positions:
            lw, lh = self.fontImages[letter].shape
            self.image[x - lw:x, y - lh:y] = 255

    def matchAllLetters(self):
        i = 1
        for element in self.orderAndFreq:
            #start = time.clock()
            letter = element[0]
            letterImage = self.fontImages[letter]
            corrCoefficient = 5 * element[1] / self.maxf / 100
            correlation = self.correlateLetter(self.image, letterImage, 254, threshold=0.88 + corrCoefficient)
            self.lettersPositions[letter] = self.getLetterPositions(correlation, letterImage)
            self.removeLetter(letter, self.lettersPositions[letter])
            #misc.imsave('data/results/' + str(i) + letter + '.png', self.image)
            i += 1
            #print(letter, " TIME : ", time.clock() - start)

    def positionsToText(self):
        positions = []
        for k, v in self.lettersPositions.items():
            for e in v:
                if (e[0] % 100 < 50):
                    positions.append((k, e[0] - e[0] % 100, e[1]))
        positions = sorted(positions, key=lambda e: (e[1], e[2]))
        text = ""
        space = self.readImage('data/sans/space.png').shape[1] * 1.7
        line = 99
        for i in range(len(positions) - 1):
            text += positions[i][0]
            if abs(positions[i][2] - positions[i + 1][2]) > space:
                text += chr(32)
            if abs(positions[i][1] - positions[i + 1][1]) >= line:
                text += '\n'
        return text


if __name__ == '__main__':
    start = time.clock()
    oceery = Recognizer(sans=True)
    oceery.matchAllLetters()
    #print("TIME AFTER MATCH  nbh: ", time.clock() - start)
    oceery.positionsToText()

    print(time.clock() - start)
