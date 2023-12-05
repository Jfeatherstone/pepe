"""
Methods to upsample or downsample an image via DNN super-resolution or pyramid scheme, respectively.

Note that the super-resolution models need to be downloaded separately, as they are not
my own original work. Four methods are compatible with this package:

- [EDSR](https://arxiv.org/pdf/1707.02921.pdf) ([implementation](https://github.com/Saafke/EDSR_Tensorflow))

- [ESPCN](https://arxiv.org/abs/1609.05158) ([implementation](https://github.com/fannymonori/TF-ESPCN))

- [FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) ([implementation](https://github.com/Saafke/FSRCNN_Tensorflow))

- [LapSRN](https://arxiv.org/abs/1710.01992) ([implementation](https://github.com/fannymonori/TF-LapSRN))

The above implementations are simply the ones that I have used in the past; there are certainly others available, which should give comparable results. I am not the creator of any of the above listed implementations either, nor am I associated in any way (other than as a user) with them.

To install any of the above models, you should place the `.pb` weight files into an appropriately named directory within the `preprocess/models/` folder. e.g. directory structure:

```
.../preprocess/
    |-- Rescale.py
    |-- ...
    `-- models/
        |-- FSRCNN/
        |   |-- FSRCNN_x2.pb
        |   |-- FSRCNN_x3.pb
        |   `-- FSRCNN_x4.pb
        `-- LapSRN/
            |-- LapSRN_x2.pb
            |-- LapSRN_x4.pb
            `-- LapSRN_x8.pb
```

Make sure that the naming conventions are consistent: the name of the directory should be the same as the first part of the weight file name, and the second part (separated by an underscore `_`) should be `x` plus the upsample factor. These conventions are important for the package to automatically detect which models are available.

"""

import numpy as np
import os

import cv2
from cv2 import dnn_superres
import pathlib
from skimage.measure import block_reduce

# The models dir should be in a directory called 'models' in the same parent directory
# as this file (likely `pepe/preprocess/models/`)
MODELS_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/models/'

# For the upscale() method, we presumably are only doing a single operation,
# but it is still good practice to hold onto the model we generate. So this
# will be assigned to a proper model after the first call to upscale()
defaultModelInstance = None
defaultModelType = 'FSRCNN'
defaultModelFactor = 2

def modelInstallHelp():
    helpString = f"""
    Note that the super-resolution models need to be downloaded separately, as they are not
    my own original work. Four methods are compatible with this package:

    - [EDSR](https://arxiv.org/pdf/1707.02921.pdf) ([implementation](https://github.com/Saafke/EDSR_Tensorflow))

    - [ESPCN](https://arxiv.org/abs/1609.05158) ([implementation](https://github.com/fannymonori/TF-ESPCN))

    - [FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) ([implementation](https://github.com/Saafke/FSRCNN_Tensorflow))

    - [LapSRN](https://arxiv.org/abs/1710.01992) ([implementation](https://github.com/fannymonori/TF-LapSRN))

    The above implementations are simply the ones that I have used in the past; there are certainly others available, which should give comparable results. I am not the creator of any of the above listed implementations either, nor am I associated in any way (other than as a user) with them.

    To install any of the above models, you should place the `.pb` weight files into an appropriately named directory within the following folder:
    {MODELS_DIR}.

    e.g. directory structure:

    ```
    .../preprocess/
        |-- Rescale.py
        |-- ...
        `-- models/
            |-- FSRCNN/
            |   |-- FSRCNN_x2.pb
            |   |-- FSRCNN_x3.pb
            |   `-- FSRCNN_x4.pb
            `-- LapSRN/
                |-- LapSRN_x2.pb
                |-- LapSRN_x4.pb
                `-- LapSRN_x8.pb
    ```
    Make sure that the naming conventions are consistent: the name of the directory should be the same as the first part of the weight file name, and the second part (separated by an underscore `_`) should be `x` plus the upsample factor. These conventions are important for the package to automatically detect which models are available.
    """

    print(helpString)

def listSuperResModels(debug=False):

    if not os.path.exists(MODELS_DIR):
        print(f'Warning: models folder ({MODELS_DIR}) does not exist!\nSee pepe.preprocess.modelInstallHelp() for more information about installing models.')
        return {}

    # Determine what models are available
    availableModels = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR + d)]
    availableModels = np.sort(availableModels)

    modelFactors = {}

    # Now see which upscale factors are available for each model
    # Each .pb file should be of the form <model_name>_x<upscale_factor>.pb
    for d in availableModels:
        weightFiles = [f for f in os.listdir(MODELS_DIR + d) if len(f) > 2 and f[-2:] == 'pb']

        if len(weightFiles) == 0:
            if debug:
                print(f'Warning: model type \'{d}\' has no valid implementations!')
            continue
        
        availableScalingFactors = []
        # Now extract the scaling factor
        for wF in weightFiles:
            try:
                # Cut off the extension, take the portion after the 'x', and cast to int
                upscaleFactor = int(wF[:-3].split('x')[-1])
                availableScalingFactors.append(upscaleFactor)
            except:
                if debug:
                    print(f'Warning: file \'{d}/{wF}\' does not conform to naming conventions.\nAll files should be named in the format <model_type>_x<upscale_factor>.pb!\nSee pepe.preprocess.modelInstallHelp() for more information about installing models.')

        modelFactors[d] = list(np.sort(availableScalingFactors))

    return modelFactors


def prepareSuperResModel(modelName, upscaleFactor):

    # Determine what models are available
    modelFactors = listSuperResModels(debug=False)
    possibleModels = list(modelFactors.keys())

    if not modelName.lower() in [p.lower() for p in possibleModels]:
        print(f'Error: requested model \'{modelName}\' not found!')
        return None

    # We could just put the keys into the dict in lowercase, but that would mean that they
    # would be displayed like that in the help screen, so I'd rather just it like this.
    for key in possibleModels:
        if modelName.lower() == key.lower():
            # Make sure the requested scaling factor is available
            if not upscaleFactor in modelFactors[key]:
                print(f'Error: requested upscale factor ({upscaleFactor}) is not available for model \'{modelName}\'! \nAvailable factors are: {modelFactors[key]}')
                return None

            # Otherwise, we are good to go, and can take this file as our model
            properName = modelName.lower()
            modelPath = MODELS_DIR + f'{key}/{key}_x{upscaleFactor}.pb'
            break

    # Since fsrcnn has a small option, which is just a smaller network, but the same structure, upscale
    # we need to adjust the proper name to just fsrcnn
    if properName == 'fsrcnn_small':
        properName = 'fsrcnn'
    
    # Create the network
    model = dnn_superres.DnnSuperResImpl_create()
    # Read in the .pb file containing the weights
    model.readModel(modelPath)
    # Set the active model type to the one we just read in
    model.setModel(properName, upscaleFactor)

    return model


def upsample(image, upscaleFactor=defaultModelFactor, modelName=defaultModelType):
    global defaultModelInstance, defaultModelType, defaultModelFactor
    # Make sure we have a (singleton) model on hand
    if defaultModelInstance is None or upscaleFactor != defaultModelFactor or modelName != defaultModelType:
        defaultModelInstance = prepareSuperResModel(modelName, upscaleFactor)
        defaultModelType = modelName
        defaultModelFactor = upscaleFactor
   
    if defaultModelInstance is None:
        return None

    # Now do the upscaling
    return defaultModelInstance.upsample(image)

def downsample(image, downscaleFactor=2):
    """
    A very simple wrapper around either `cv2.pyrDown()` or
    `skimage.measure.block_reduce()`, depending on the desired
    downscale factor.

    Parameters
    ----------

    image : np.ndarray[H,W]
        The image to be downsampled

    downscaleFactor : int
        The factor by which to downsample the image.

    Returns
    -------

    dsImage : np.ndarray[H/downscaleFactor,W/downscaleFactor]
        Downsampled image.
    """
    if downscaleFactor == 2:
        return cv2.pyrDown(image)
    else:
        return block_reduce(image, downscaleFactor)
    
