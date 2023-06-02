import numpy as np

import pathlib

from pepe.preprocess import checkImageType
from pepe.tracking import convCircle, houghCircle
from pepe.utils import preserveOrderArgsort

DATA_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/test_data'

def test_convCircle_track():
    """
    Test convolutional tracking on generated
    test images.
    """
    for i in range(3):
        image = checkImageType(f'{DATA_DIR}/test_circles_{i+1}.bmp').astype(np.uint8)
        radius = 50
        tolerance = .10 # 10% tolerance of the radius

        # Parameters aren't too important
        particleCenters, particleRadii = convCircle(image, radius, negativeHalo=True,
                                                    intensitySoftmax=1.8, intensitySoftmin=.6, peakDownsample=1,
                                                    offscreenParticles=True, radiusTolerance=2, minPeakPrevalence=.4)

        with open(f'{DATA_DIR}/test_circles_{i+1}_centers.npy', 'rb') as f:
            expectedCenters = np.load(f)

        assert len(expectedCenters) == len(particleCenters), f"Incorrect number of particles detected in test image {i+1}: {len(particleCenters)} vs {len(expectedCenters)}."

        sortOrder = preserveOrderArgsort(particleCenters, expectedCenters)

        difference = np.sqrt(np.sum((particleCenters - expectedCenters[sortOrder])**2, axis=-1))

        percentOfRadius = difference / radius
        assert not True in [p > tolerance for p in percentOfRadius], f"Particle centers not within {tolerance*100}% of expected value in test image {i+1}."
