"""
Defines the Trial object to assist with analyzing the output from `pepe.auto.forceSolve()`.
"""

import numpy as np
import inspect
import pickle
import os
from PIL import Image

from pepe.auto import forceSolveArgDTypes
from pepe.analysis import forceOptimizeArgDTypes
from pepe.tracking import circleTrackArgDTypes
from pepe.utils import parseList
from pepe.preprocess import checkImageType

class Trial():

    def __init__(self, datasetDir=None, fArr=None, aArr=None, bArr=None, cArr=None, rArr=None, thArr=None):

        if datasetDir is not None:
            self.datasetPath = os.path.abspath(datasetDir) + '/'

            if not os.path.exists(self.datasetPath):
                raise Exception(f'Directory {self.datasetPath} does not exist!')

            # Check to make sure all of the pickle files exist
            fileNames = ['forces', 'alphas', 'betas', 'centers', 'radii', 'angles']
            fileExists = {}
            for f in fileNames:
                fileExists[f] = os.path.exists(self.datasetPath + f + '.pickle')

            # Make sure we have at least one of the files
            if np.sum(np.int64(list(fileExists.values()))) == 0:
                raise Exception(f'Directory {self.datasetPath} doesn\'t contain any data... Are you sure this is the correct directory?')

            # If we are only missing one or two, we can just print that
            for f in fileNames:
                if not fileExists[f]:
                    print(f'Warning: dataset {datasetDir} is missing {f}.pickle file!\n {f[0].upper()}{f[1:]} will not be available for this object.')

            # Load in all of the pickle files
            with open(self.datasetPath + 'forces.pickle', 'rb') as f:
                if fileExists['forces']:
                    self.forceArr = pickle.load(f)
                else:
                    self.forceArr = None

            with open(self.datasetPath + 'betas.pickle', 'rb') as f:
                if fileExists['betas']:
                    self.betaArr = pickle.load(f)
                else:
                    self.betaArr = None

            with open(self.datasetPath + 'alphas.pickle', 'rb') as f:
                if fileExists['alphas']:
                    self.alphaArr = pickle.load(f)
                else:
                    self.alphaArr = None

            with open(self.datasetPath + 'centers.pickle', 'rb') as f:
                if fileExists['centers']:
                    self.centerArr = pickle.load(f)
                else:
                    self.centerArr = None

            with open(self.datasetPath + 'radii.pickle', 'rb') as f:
                if fileExists['radii']:
                    self.radiusArr = pickle.load(f)
                else:
                    self.radiusArr = None

            with open(self.datasetPath + 'angles.pickle', 'rb') as f:
                if fileExists['angles']:
                    self.angleArr = pickle.load(f)
                else:
                    self.angleArr = None

            # And try to load in the settings
            settingsFile = self.datasetPath + 'readme.txt'
            self.settings = {}
            settingDTypes = forceSolveArgDTypes.copy()
            settingDTypes.update(circleTrackArgDTypes)
            settingDTypes.update(forceOptimizeArgDTypes)

            if os.path.exists(settingsFile):
                    fileObj = open(settingsFile, 'r')
                    for line in fileObj:
                        # Check each line and see if it looks like a dictionary value
                        split = line.split(':')
                        # Read settings into the master settings file
                        if len(split) == 2 and split[0].strip() in settingDTypes.keys():
                            # Cast to the type of the value already in the dict
                            if split[1].strip() == 'None':
                                self.settings[split[0].strip()] = None
                            else:
                                if '[' in split[1]:
                                    self.settings[split[0].strip()] = parseList(split[1].strip(), dtype=settingDTypes[split[0].strip()])
                                else:
                                    # Bools need a special condition
                                    if settingDTypes[split[0].strip()] is bool:
                                        val = split[1].strip() == 'True'
                                    else:
                                        val = settingDTypes[split[0].strip()](split[1].strip())

                                    self.settings[split[0].strip()] = val


            # Load in the particle identities too, since that is often very useful
            if self.settings["genFitReport"]:
                self.identitiesImage = Image.fromarray(checkImageType(self.datasetPath + 'FitReport_src/particle_identities.png'))
            else:
                self.identitiesImage = None
        else:
            # If we are explicitly passed arrays, we just assign those
            self.forceArr = fArr
            self.alphaArr = aArr
            self.betaArr = bArr
            self.centerArr = cArr
            self.radiusArr = rArr
            self.angleArr = thArr

            self.identitiesImage = None
            self.datasetPath = None

        # Assign some constants
        if self.forceArr is not None:
            self.numParticles = len(self.forceArr)
            self.numForces = [len(f) for f in self.forceArr]
            self.numTimesteps = [f[0].shape[-1] for f in self.forceArr if len(f) > 0][0]


    def particleNear(self, point, timestep=0):
        distances = np.sum((self.centerArr[:,timestep,:] - np.array(point))**2, axis=-1)
        return np.argmin([d for d in distances if not np.isnan(d)])


    def averageForcePositions(self, particleIndex):
        averageBetaArr = np.array([np.nanmean(self.betaArr[particleIndex][i,:]) for i in range(self.numForces[particleIndex])])
        return averageBetaArr
                                    

    def __str__(self):
        return self.datasetPath.split('/')[-2] if self.datasetPath is not None else super.__str__()
