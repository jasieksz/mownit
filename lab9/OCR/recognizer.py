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

    def readImage(self, path):
        result = ndimage.imread(path, mode='I')
        return result

    def invertImage(self, image):
        return 255 - image

    def showImage(self, image, size):
        plt.figure(figsize=(size, size))
        plt.imshow(image)
        plt.show()

    def getSansPathDict(self, sans):
        path = 'data/sans/' if sans else 'data/serif/'
        letters = [chr(i) for i in range(97, 123)]
        sansPathDict = {}
        for l in letters:
            sansPathDict[l] = path + l + '.png'
        return sansPathDict

    def getLettersImages(self, sans):
        result = {}
        paths = self.getSansPathDict(sans)
        letters = [chr(i) for i in range(97, 123)]
        images = [self.invertImage(self.readImage(v)) for k, v in paths]
        for letter, i in zip(letters, range(len(letters))):
            result[letter] = images[i]
        return result

    def correlateLetter(self, img, letter, color, threshold=0.82):
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
            self.image[x - lw:x, y-lh:y] = 0

    def matchAllLetters(self):
        for letter, letterImage in self.fontImages:
            start = time.clock()
            correlation = self.correlateLetter(self.image, letter, 254, threshold=0.85)
            self.lettersPositions[letter] = self.getLetterPositions(correlation, letter)
            if (letter == 'a'):
                self.showImage(correlation)
                print(self.lettersPositions[letter])
            self.removeLetter(letter, self.lettersPositions[letter])
            print(letter, " TIME : ", time.clock() - start)


if __name__ == '__main__':
    start = time.clock()
    ocery = Recognizer()
    ocery.matchAllLetters()
    ocery.showImage(ocery.image, 15)
    print(time.clock() - start)