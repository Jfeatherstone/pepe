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
        # This is quite a large tolerance if we were actually doing some analysis, but here
        # we just want to make sure the algorithm works as usual.
        tolerance = 10 # pixels 

        # Parameters aren't too important
        particleCenters, particleRadii = convCircle(image, radius, negativeHalo=True,
                                                    intensitySoftmax=2, intensitySoftmin=.5, peakDownsample=1,
                                                    offscreenParticles=True, radiusTolerance=None, minPeakPrevalence=.4)

        with open(f'{DATA_DIR}/test_circles_{i+1}_centers.npy', 'rb') as f:
            expectedCenters = np.load(f)

        assert len(expectedCenters) == len(particleCenters), f"Incorrect number of particles detected in test image {i+1}: {len(particleCenters)} vs {len(expectedCenters)}."

        sortOrder = preserveOrderArgsort(particleCenters, expectedCenters)

        difference = np.sqrt(np.sum((particleCenters - expectedCenters[sortOrder])**2, axis=-1))

        assert not (True in [d > tolerance for d in difference]), f"Particle centers not within {tolerance} pixels of expected value in test image {i+1}.\n\nExpected centers:{expectedCenters[sortOrder]}\n\nFound centers:{particleCenters}"
