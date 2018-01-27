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
        self.maxf = 63
        self.orderAndFreq = [('x', 1),
                     ('z', 1),
                     ('w', 4),
                     ('y', 4),
                     ('j', 7),
                     ('f', 8),
                     ('v', 8),
                     ('k', 9),
                     ('g', 10),
                     ('p', 11),
                     ('d', 12),
                     ('m', 13),
                     ('b', 14),
                     ('n', 21),
                     ('u', 21),
                     ('h', 26),
                     ('a', 27),
                     ('t', 28),
                     ('i', 30),
                     ('q', 32),
                     ('r', 34),
                     ('l', 39),
                     ('o', 41),
                     ('s', 45),
                     ('e', 51),
                     ('c', 63)]

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
            corrCoefficient = 5 * element[1] / self.maxf / 100
            correlation = self.correlateLetter(self.image, letterImage, 254, threshold=0.8 + corrCoefficient)
            self.lettersPositions[letter] = self.getLetterPositions(correlation, letterImage)
            self.removeLetter(letter, self.lettersPositions[letter])
            misc.imsave('data/results/' + letter + '.png', self.image)
            print(letter, " TIME : ", time.clock() - start)


if __name__ == '__main__':
    start = time.clock()
    oceery = Recognizer()
    oceery.matchAllLetters()
    print("TIME AFTER MATCH : ", time.clock() - start)
    np.save('data/result_norm.npy', oceery.lettersPositions)

    # dic = np.load('data/result.npy')
    # frq = [(chr(i), dic.item()[chr(i)]) for i in range(97, 123)]
    # print(len(dic.item()['a']))
    # print(sorted([(e[0], len(e[1])) for e in frq], key=lambda e: e[1]))

    # dic2 = np.load('data/result_norm.npy')
    # dic2 = dic2.item()
    # dic2 = [(k, len(v)) for k,v in dic2.items()]
    # print(dic2)
    print(time.clock() - start)
