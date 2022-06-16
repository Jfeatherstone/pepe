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
        """
        Helper class for reading in the output of `pepe.auto.forceSolve()`.

        Parameters
        ----------

        datasetDir : str
            Path to the directory containing readme and pickle files.

        fArr : array_like or None
            Force array, if Trial object is to be constructed manually.

        aArr : array_like or None
            Alpha array, if Trial object is to be constructed manually.

        bArr : array_like or None
            Beta array, if Trial object is to be constructed manually.

        cArr : array_like or None
            Center array, if Trial object is to be constructed manually.

        rArr : array_like or None
            Radius array, if Trial object is to be constructed manually.

        thArr : array_like or None
            Angle array, if Trial object is to be constructed manually.
        """

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
            if fileExists['forces']:
                with open(self.datasetPath + 'forces.pickle', 'rb') as f:
                    self.forceArr = pickle.load(f)
            else:
                self.forceArr = None

            if fileExists['betas']:
                with open(self.datasetPath + 'betas.pickle', 'rb') as f:
                    self.betaArr = pickle.load(f)
            else:
                self.betaArr = None

            if fileExists['alphas']:
                with open(self.datasetPath + 'alphas.pickle', 'rb') as f:
                    self.alphaArr = pickle.load(f)
            else:
                self.alphaArr = None

            if fileExists['centers']:
                with open(self.datasetPath + 'centers.pickle', 'rb') as f:
                    self.centerArr = pickle.load(f)
            else:
                self.centerArr = None

            if fileExists['radii']:
                with open(self.datasetPath + 'radii.pickle', 'rb') as f:
                    self.radiusArr = pickle.load(f)
            else:
                self.radiusArr = None

            if fileExists['angles']:
                with open(self.datasetPath + 'angles.pickle', 'rb') as f:
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
            if 'genFitReport' in self.settings.keys() and self.settings["genFitReport"]:
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

    def fixAllNan(self, interpolateForce=True):
        """
        Replaces all nan values in all data arrays with physically inspired
        values.

        Enacts the following behavior on each data type:

            forceArr: replaces with 0 or interpolates
            angleArr: interpolates
            alphaArr: interpolates
            betaArr: interpolates
            radiusArr: interpolates
            centerArr: interpolates
        """
        for i in range(self.numParticles):

            if not interpolateForce:
                # Forces
                # Just replace all nan values with 0
                self.forceArr[i] = np.where(np.isnan(self.forceArr[i]), 0, self.forceArr[i])

            # Interpolate for the rest
            # In theory, when one is nan, the others should be as well, but
            # this isn't guaranteed, so it is safest just to go through each
            # separately

            # Angles
            angleNanIndices = np.where(np.isnan(self.angleArr[i]))[0]

            for j in range(len(angleNanIndices)):
                previousGoodPoint = [ind for ind in np.arange(angleNanIndices[j]) if not ind in angleNanIndices]
                previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                nextGoodPoint = [ind for ind in np.arange(angleNanIndices[j], self.numTimesteps) if not ind in angleNanIndices]
                nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1
               
                # If one of the points before or after doesn't exist, just copy over the
                # one that does. We should never have a case in which both points don't
                # exist, because that would mean we have an entire array of nan values
                if previousGoodPoint < 0:
                    self.angleArr[i][angleNanIndices[j]] = self.angleArr[i][nextGoodPoint]
                elif nextGoodPoint < 0:
                    self.angleArr[i][angleNanIndices[j]] = self.angleArr[i][previousGoodPoint]
                else:
                    self.angleArr[i][angleNanIndices[j]] = (self.angleArr[i][previousGoodPoint] * (angleNanIndices[j] - previousGoodPoint) + self.angleArr[i][nextGoodPoint] * (nextGoodPoint - angleNanIndices[j])) / (nextGoodPoint - previousGoodPoint)


            # Centers
            centerNanIndices = np.where(np.isnan(self.centerArr[i][:,0]))[0]

            for j in range(len(centerNanIndices)):
                previousGoodPoint = [ind for ind in np.arange(centerNanIndices[j]) if not ind in centerNanIndices]
                previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                nextGoodPoint = [ind for ind in np.arange(centerNanIndices[j], self.numTimesteps) if not ind in centerNanIndices]
                nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1
              
                # If one of the points before or after doesn't exist, just copy over the
                # one that does. We should never have a case in which both points don't
                # exist, because that would mean we have an entire array of nan values
                if previousGoodPoint < 0:
                    self.centerArr[i][centerNanIndices[j]] = self.centerArr[i][nextGoodPoint]
                elif nextGoodPoint < 0:
                    self.centerArr[i][centerNanIndices[j]] = self.centerArr[i][previousGoodPoint]
                else:
                    # Loop over y and x
                    for k in range(2):
                        self.centerArr[i][angleNanIndices[j]][k] = (self.centerArr[i][previousGoodPoint][k] * (centerNanIndices[j] - previousGoodPoint) + self.centerArr[i][nextGoodPoint][k] * (nextGoodPoint - centerNanIndices[j])) / (nextGoodPoint - previousGoodPoint)

            # Radii
            radiusNanIndices = np.where(np.isnan(self.radiusArr[i]))[0]

            for j in range(len(radiusNanIndices)):
                previousGoodPoint = [ind for ind in np.arange(radiusNanIndices[j]) if not ind in radiusNanIndices]
                previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                nextGoodPoint = [ind for ind in np.arange(radiusNanIndices[j], self.numTimesteps) if not ind in radiusNanIndices]
                nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1

                # If one of the points before or after doesn't exist, just copy over the
                # one that does. We should never have a case in which both points don't
                # exist, because that would mean we have an entire array of nan values
                if previousGoodPoint < 0:
                    self.radiusArr[i][radiusNanIndices[j]] = self.radiusArr[i][nextGoodPoint]
                elif nextGoodPoint < 0:
                    self.radiusArr[i][radiusNanIndices[j]] = self.radiusArr[i][previousGoodPoint]
                else:  
                    self.radiusArr[i][angleNanIndices[j]] = (self.radiusArr[i][previousGoodPoint] * (radiusNanIndices[j] - previousGoodPoint) + self.radiusArr[i][nextGoodPoint] * (nextGoodPoint - radiusNanIndices[j])) / (nextGoodPoint - previousGoodPoint)

            # For the last two, we have to iterate over each force
            for j in range(self.numForces[i]):

                # Alphas
                alphaNanIndices = np.where(np.isnan(self.alphaArr[i][j]))[0]

                for k in range(len(alphaNanIndices)):
                    previousGoodPoint = [ind for ind in np.arange(alphaNanIndices[k]) if not ind in alphaNanIndices]
                    previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                    nextGoodPoint = [ind for ind in np.arange(alphaNanIndices[k], self.numTimesteps) if not ind in alphaNanIndices]
                    nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1

                    # If one of the points before or after doesn't exist, just copy over the
                    # one that does. We should never have a case in which both points don't
                    # exist, because that would mean we have an entire array of nan values
                    if previousGoodPoint < 0:
                        self.alphaArr[i][j][alphaNanIndices[k]] = self.alphaArr[i][j][nextGoodPoint]
                    elif nextGoodPoint < 0:
                        self.alphaArr[i][j][alphaNanIndices[k]] = self.alphaArr[i][j][previousGoodPoint]
                    else:   
                        self.alphaArr[i][j][alphaNanIndices[k]] = (self.alphaArr[i][j][previousGoodPoint] * (alphaNanIndices[k] - previousGoodPoint) + self.alphaArr[i][j][nextGoodPoint] * (nextGoodPoint - alphaNanIndices[k])) / (nextGoodPoint - previousGoodPoint)

                # Betas
                betaNanIndices = np.where(np.isnan(self.betaArr[i][j]))[0]

                for k in range(len(betaNanIndices)):
                    previousGoodPoint = [ind for ind in np.arange(betaNanIndices[k]) if not ind in betaNanIndices]
                    previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                    nextGoodPoint = [ind for ind in np.arange(betaNanIndices[k], self.numTimesteps) if not ind in betaNanIndices]
                    nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1
                    
                    # If one of the points before or after doesn't exist, just copy over the
                    # one that does. We should never have a case in which both points don't
                    # exist, because that would mean we have an entire array of nan values
                    if previousGoodPoint < 0:
                        self.betaArr[i][j][betaNanIndices[k]] = self.betaArr[i][j][nextGoodPoint]
                    elif nextGoodPoint < 0:
                        self.betaArr[i][j][betaNanIndices[k]] = self.betaArr[i][j][previousGoodPoint]
                    else:   
                        self.betaArr[i][j][betaNanIndices[k]] = (self.betaArr[i][j][previousGoodPoint] * (alphaNanIndices[k] - previousGoodPoint) + self.alphaArr[i][j][nextGoodPoint] * (nextGoodPoint - alphaNanIndices[k])) / (nextGoodPoint - previousGoodPoint)

                # Forces
                if interpolateForce:
                    forceNanIndices = np.where(np.isnan(self.forceArr[i][j]))[0]

                    for k in range(len(forceNanIndices)):
                        previousGoodPoint = [ind for ind in np.arange(forceNanIndices[k]) if not ind in forceNanIndices]
                        previousGoodPoint = previousGoodPoint[-1] if len(previousGoodPoint) > 0 else -1
                        nextGoodPoint = [ind for ind in np.arange(forceNanIndices[k], self.numTimesteps) if not ind in forceNanIndices]
                        nextGoodPoint = nextGoodPoint[0] if len(nextGoodPoint) > 0 else -1
                        
                        # If one of the points before or after doesn't exist, just copy over the
                        # one that does. We should never have a case in which both points don't
                        # exist, because that would mean we have an entire array of nan values
                        if previousGoodPoint < 0 or nextGoodPoint < 0:
                            self.forceArr[i][j][forceNanIndices[k]] = 0
                        else:   
                            self.forceArr[i][j][forceNanIndices[k]] = (self.forceArr[i][j][previousGoodPoint] * (forceNanIndices[k] - previousGoodPoint) + self.forceArr[i][j][nextGoodPoint] * (nextGoodPoint - forceNanIndices[k])) / (nextGoodPoint - previousGoodPoint)



    def particleNear(self, point, timestep=0):
        """
        Identifies the index of the particle closest to the provided point
        at the provided timestep.
        """

        distances = np.sum((self.centerArr[:,timestep,:] - np.array(point))**2, axis=-1)
        return np.argmin([d for d in distances if not np.isnan(d)])


    def averageForcePositions(self, particleIndex):
        """
        Calculates the time-averaged beta values of forces for the provided
        particle.
        """
        #averageBetaArr = np.array([np.nanmean(self.betaArr[particleIndex][i,:]) for i in range(self.numForces[particleIndex])])
        averageBetaArr = np.array([np.nanmedian(self.betaArr[particleIndex][i,:]) for i in range(self.numForces[particleIndex])])
        # Now correct for periodicity
        #return np.array([ab if ab <= np.pi else ab - 2*np.pi for ab in averageBetaArr])
        return averageBetaArr
                                    

    def __str__(self):
        return self.datasetPath.split('/')[-2] if self.datasetPath is not None else super.__str__()
