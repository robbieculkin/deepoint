import numpy as np

def random_uniform(height, width):
    return 255*np.random.rand(height, width)

def salt_and_pepper(height, width):
    return 255*np.random.randint(0,2, size=(height, width))

# grabbed this code fromg ghub
import math, random
class AdvancedTextures:
    """
    Texture generation using Perlin noise
    """ 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gradientNumber = 256

        self.grid = [[]]
        self.gradients = []
        self.permutations = []
        self.img = {}

        self.__generateGradientVectors()
        self.__normalizeGradientVectors()
        self.__generatePermutationsTable()

    def __call__(self, func):
        self.makeTexture(getattr(self, func))
        pixels = np.zeros((self.x, self.y))
        for i in range(0, self.x):
            for j in range(0, self.y):
                c = self.img[i, j]
                pixels[i, j] = c
        return pixels

    def __generateGradientVectors(self):
        for i in range(self.gradientNumber):
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x * x + y * y < 1:
                    self.gradients.append([x, y])
                    break

    def __normalizeGradientVectors(self):
        for i in range(self.gradientNumber):
            x, y = self.gradients[i][0], self.gradients[i][1]
            length = math.sqrt(x * x + y * y)
            self.gradients[i] = [x / length, y / length]

    # The modern version of the Fisher-Yates shuffle
    def __generatePermutationsTable(self):
        self.permutations = [i for i in range(self.gradientNumber)]
        for i in reversed(range(self.gradientNumber)):
            j = random.randint(0, i)
            self.permutations[i], self.permutations[j] = \
                self.permutations[j], self.permutations[i]

    def getGradientIndex(self, x, y):
        return self.permutations[(x + self.permutations[y % self.gradientNumber]) % self.gradientNumber]

    def perlinNoise(self, x, y):
        qx0 = int(math.floor(x))
        qx1 = qx0 + 1

        qy0 = int(math.floor(y))
        qy1 = qy0 + 1

        q00 = self.getGradientIndex(qx0, qy0)
        q01 = self.getGradientIndex(qx1, qy0)
        q10 = self.getGradientIndex(qx0, qy1)
        q11 = self.getGradientIndex(qx1, qy1)

        tx0 = x - math.floor(x)
        tx1 = tx0 - 1

        ty0 = y - math.floor(y)
        ty1 = ty0 - 1

        v00 = self.gradients[q00][0] * tx0 + self.gradients[q00][1] * ty0
        v01 = self.gradients[q01][0] * tx1 + self.gradients[q01][1] * ty0
        v10 = self.gradients[q10][0] * tx0 + self.gradients[q10][1] * ty1
        v11 = self.gradients[q11][0] * tx1 + self.gradients[q11][1] * ty1

        wx = tx0 * tx0 * (3 - 2 * tx0)
        v0 = v00 + wx * (v01 - v00)
        v1 = v10 + wx * (v11 - v10)

        wy = ty0 * ty0 * (3 - 2 * ty0)
        return (v0 + wy * (v1 - v0)) * 0.5 + 1

    def makeTexture(self, texture = None):
        if texture is None:
            texture = self.cloud

        noise = {}
        max = min = None
        for i in range(self.x):
            for j in range(self.y):
                value = texture(i, j)
                noise[i, j] = value
                
                if max is None or max < value:
                    max = value

                if min is None or min > value:
                    min = value

        for i in range(self.x):
            for j in range(self.y):
                self.img[i, j] = (int) ((noise[i, j] - min) / (max - min) * 255 )

    def fractalBrownianMotion(self, x, y, func):
        octaves = 12
        amplitude = 1.0
        x_frequency = 1.0 / self.x
        y_frequency = 1.0 / self.y
        persistence = 0.5
        value = 0.0
        for k in range(octaves):
            value += func(x * x_frequency, y * y_frequency) * amplitude
            x_frequency *= 2
            y_frequency *= 2
            amplitude *= persistence
        return value

    def cloud(self, x, y, func = None):
        if func is None:
            func = self.perlinNoise

        return self.fractalBrownianMotion(8 * x, 8 * y, func)

    def wood(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        x_frequency = 1.0 / self.x
        y_frequency = 1.0 / self.y
        n = noise(4 * x * x_frequency, 4 * y * y_frequency) * 10
        return n - int(n)

    def marble(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        frequency = 1.0 / self.x
        n = self.fractalBrownianMotion(8 * x, 8 * y, self.perlinNoise)
        return (math.sin(16 * x * frequency + 4 * (n - 0.5)) + 1) * 0.5
