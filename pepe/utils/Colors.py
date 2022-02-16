import numpy as np

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def genRandomColors(size, seed=21):
    np.random.seed(seed)

    randomColors = [f"#{rgb_to_hex(tuple(np.random.choice(range(256), size=3).flatten()))}" for i in range(size)]

    return randomColors
