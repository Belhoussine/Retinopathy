from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import gridspec
import matplotlib.pyplot as plt
from random import randint


# Convert image to matrix
def toMatrix(image):
    return np.copy(np.asarray(image))

# Convert matrix to image
def toImage(matrix):
    return Image.fromarray(matrix.astype('uint8'))


class RetinalFundusImage:
    # Constructor
    def __init__(self, path, id, dims=(2, 2), plot=True):
        self.id = id
        self.path = path
        self.image = self.getImage()
        self.red = self.getChannel(channel='r', asMatrix=True)
        self.green = self.getChannel(channel='g', asMatrix=True)
        self.blue = self.getChannel(channel='b', asMatrix=True)
        if(plot):
            self.plotConfig(dims)

    # Get image as a PIL.Image or matrix
    def getImage(self, asMatrix=False):
        image = Image.open(path)
        if(image.size != (2592, 1728)):
            image = image.resize(((2592, 1728)))
            # raise ValueError("Image size is not correct.")
        image = image.crop((450, 0, 2150, image.size[1]))
        imageMatrix = np.copy(np.asarray(image))
        return imageMatrix if asMatrix else image

    # Get RGB channels as a PIL.Image or matrix
    def getChannels(self, asMatrix=False):
        channels = self.image.split()
        channelsMatrix = list(map(toMatrix, channels))
        return channelsMatrix if asMatrix else channels

    # Get single RGB channel
    def getChannel(self, asMatrix=False, channel='g'):
        channels = self.getChannels()
        channelsMatrix = self.getChannels(asMatrix=True)
        if(channel.lower() == 'r'):
            return channelsMatrix[0] if asMatrix else channels[0]
        elif(channel.lower() == 'g'):
            return channelsMatrix[1] if asMatrix else channels[1]
        elif(channel.lower() == 'b'):
            return channelsMatrix[2] if asMatrix else channels[2]

    # Apply CLAHE on a specific channel
    def applyCLAHE(self, channel='g', clipValue=2.0):
        self.claheApplied = True
        clahe = cv.createCLAHE(clipLimit=clipValue, tileGridSize=(8, 8))
        if channel == 'r':
            self.red = clahe.apply(self.red)
        elif channel == 'g':
            self.green = clahe.apply(self.green)
        elif channel == 'b':
            self.blue = clahe.apply(self.blue)

    # Apply median filter
    def applyMedianFilter(self, channel='g', kernelSize=3):
        self.medianApplied = True
        if channel == 'r':
            self.red = cv.medianBlur(self.red, kernelSize)
        elif channel == 'g':
            self.green = cv.medianBlur(self.green, kernelSize)
        elif channel == 'b':
            self.blue = cv.medianBlur(self.blue, kernelSize)

    # Apply thresholding
    def applyThresholding(self, channel='g', blockSize=99, C=24):
        self.thresholdApplied = True
        if channel == 'r':
            self.red = cv.adaptiveThreshold(
                self.red, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, C)
        elif channel == 'g':
            self.green = cv.adaptiveThreshold(
                self.green, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, C)
        elif channel == 'b':
            self.blue = cv.adaptiveThreshold(
                self.blue, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, C)

    # Apply morphological cleaning
    def applyMorphologicalCleaning(self, channel='g', kernel=(3, 3), iterations=2):
        self.morphologicalCleaningApplied = True
        if channel == 'r':
            self.red = cv.morphologyEx(
                self.red, cv.MORPH_OPEN, kernel, iterations=iterations)
            self.applyMedianFilter(channel='r')
        elif channel == 'g':
            self.green = cv.morphologyEx(
                self.green, cv.MORPH_OPEN, kernel, iterations=iterations)
            self.applyMedianFilter(channel='g')

        elif channel == 'b':
            self.blue = cv.morphologyEx(
                self.blue, cv.MORPH_OPEN, kernel, iterations=iterations)
            self.applyMedianFilter(channel='b')

    # Fill blank areas with surrounding white pixels
    def applyMorphologicalClosing(self, channel='g', iterations=3, kernel=(5, 5)):
        for it in range(iterations):
            if channel == 'r':
                self.red = cv.morphologyEx(
                    self.red, cv.MORPH_CLOSE, kernel, iterations=5)

            elif channel == 'g':
                self.green = cv.morphologyEx(
                    self.green, cv.MORPH_CLOSE, kernel, iterations=5)

            elif channel == 'b':
                self.blue = cv.morphologyEx(
                    self.blue, cv.MORPH_CLOSE, kernel, iterations=3)

    # Check if out of bounds
    def outOfBounds(self, image, i, j):
        x = len(image)-1
        y = len(image[0])-1
        return i > x or i < 0 or j > y or j < 0

    # Count number of white neighbours to a pixel
    def getNeighbours(self, image, i, j, vis, radius=30):
        if(radius == 0):
            return False
        if(self.outOfBounds(image, i, j) or (i, j) in vis):
            return True
        vis.append((i, j))
        if(image[i][j] != 0):
            return self.getNeighbours(image, i+1, j, vis, radius=radius-1) and self.getNeighbours(image, i-1, j, vis, radius=radius-1) and self.getNeighbours(image, i, j+1, vis, radius=radius-1) and self.getNeighbours(image, i, j-1, vis, radius=radius-1)
        return True

    # Remove pixel islands
    def removeIsolatedPixels(self, image, maxPixels=30):
        skip = [[False for i in range(len(image[0]))]
                for j in range(len(image))]
        for i in range(len(image)):
            for j in range(len(image[0])):
                pixel = image[i][j]
                vis = []
                if(pixel != 0 and not skip[i][j]):
                    isolated = self.getNeighbours(
                        image, i, j, vis, radius=maxPixels)
                    for x, y in vis:
                        if isolated:
                            image[x][y] = 0
                        else:
                            skip[x][y] = True

    # Set up plot configuration
    def plotConfig(self, dims):
        rows, cols = dims
        self.index = 0
        self.fig = plt.figure(figsize=(50, 50))
        self.gs = gridspec.GridSpec(rows, cols, width_ratios=[
                                    1, 1], wspace=0.1, hspace=0.2, top=0.95, bottom=0.05, left=0.17, right=0.845)
        # Booleans
        self.claheApplied = False
        self.medianApplied = False
        self.thresholdApplied = False
        self.morphologicalCleaningApplied = False

    # Show original image
    def drawOriginal(self, grey=True):
        sub = self.fig.add_subplot(self.gs[self.index])
        sub.title.set_text('Original image')
        self.index += 1
        plt.imshow(self.image)

    # Draw a specific channel
    def drawChannel(self, channel='g'):
        image = self.green if channel == 'g' else self.red if channel == 'r' else self.blue
        title = 'Green channel' if channel == 'g' else 'Red channel' if channel == 'r' else 'Blue channel'
        title += ' after Morphological Cleaning' if self.morphologicalCleaningApplied else ' after Thresholding' if self.thresholdApplied else ' after Median filter' if self.medianApplied else ' after CLAHE' if self.claheApplied else ''
        sub = self.fig.add_subplot(self.gs[self.index])
        sub.title.set_text(title)
        self.index += 1
        plt.imshow(image, cmap='gray')

    # Show drawn images
    def show(self):
        plt.show()


def preProcess(retinalFundus):
    retinalFundus.drawOriginal()
    retinalFundus.drawChannel()
    # im = toImage(retinalFundus.green)
    # im.save(f"./figure2.jpg")

    retinalFundus.applyCLAHE(clipValue=2.5)
    # im = toImage(retinalFundus.green)
    # im.save(f"./CLAHE.jpg")
    retinalFundus.applyMedianFilter()
    # im = toImage(retinalFundus.green)
    # im.save(f"./Median.jpg")


def segment(retinalFundus):
    retinalFundus.applyThresholding(blockSize=53, C=11)
    retinalFundus.drawChannel()

    # im = toImage(retinalFundus.green)
    # im.save(f"./MeanC.jpg")


def postProcess(retinalFundus):
    # retinalFundus.show()
    retinalFundus.applyMorphologicalCleaning(kernel=(5, 5), iterations=1)
    retinalFundus.applyMorphologicalClosing(iterations=2)
    retinalFundus.removeIsolatedPixels(retinalFundus.green, maxPixels=100)
    retinalFundus.drawChannel()
    retinalFundus.show()


def getPath(i):
    return f"../Dataset/DR/{i}.png"


def randomPath(n):
    seed = randint(1, n)
    return getPath(seed), seed


if __name__ == '__main__':
    path, i = randomPath(50)
    retinalFundus = RetinalFundusImage(path, i)
    preProcess(retinalFundus)
    segment(retinalFundus)
    postProcess(retinalFundus)
