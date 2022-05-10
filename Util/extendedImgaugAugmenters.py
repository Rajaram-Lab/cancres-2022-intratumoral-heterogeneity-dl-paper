"""
This file adds HEDAdjust the functionality of the imgaug library to enable the HEDAdjust functionality. 
This idea comes from the following paper ::

H&E stain augmentation improves generalization of convolutional networks for histopathological mitosis detection.
by David Tellez, Maschenka Balkenhol, Nico Karssemeijer, Geert Litjens, Jeroen van der Laak and Francesco Ciompi
# https://geertlitjens.nl/publication/tell-18-a/tell-18-a.pdf

 Copyright (C) 2021, Rajaram Lab - UTSouthwestern 
    
    This file is part of cancres-2022-intratumoral-heterogeneity-dl-paper.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with cancres-2022-intratumoral-heterogeneity-dl-paper.  If not, see <http://www.gnu.org/licenses/>.
    
    Paul Acosta, Vipul Jarmale 2022
"""
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from imgaug.augmenters import Augmenter

# define new classes where each class is a subclass of ia.augmenters.Augmenter

class HEDadjust(Augmenter):
    
    def __init__(self, hAlpha=(0.95, 1.05), eAlpha=(0.95, 1.05), rAlpha=(0.95, 1.05), hBeta=(-0.05, 0.05), eBeta=(-0.05, 0.05), rBeta=(-0.05, 0.05), name=None, deterministic=False, random_state=None):
        """
            HEDadjust for images using a uniform distribution between ranges specified
            The value for each of hAlpha, eAlpha, rAlpha, hBeta, eBeta, rBeta is sampled per image
        """
        super(HEDadjust, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        # check for correct ranges for Alpha and Beta
        assert hAlpha[0] <= hAlpha[1], "ERROR: hAlpha[0] > hAlpha[1]"
        assert eAlpha[0] <= eAlpha[1], "ERROR: eAlpha[0] > eAlpha[1]"
        assert rAlpha[0] <= rAlpha[1], "ERROR: rAlpha[0] > rAlpha[1]"
        assert hBeta[0] <= hBeta[1], "ERROR: hBeta[0] > hBeta[1]"
        assert eBeta[0] <= eBeta[1], "ERROR: eBeta[0] > eBeta[1]"
        assert rBeta[0] <= rBeta[1], "ERROR: rBeta[0] > rBeta[1]"

        self.hAlpha_min, self.hAlpha_max = hAlpha[0], hAlpha[1]
        self.eAlpha_min, self.eAlpha_max = eAlpha[0], eAlpha[1]
        self.rAlpha_min, self.rAlpha_max = rAlpha[0], rAlpha[1]
        self.hBeta_min, self.hBeta_max = hBeta[0], hBeta[1]
        self.eBeta_min, self.eBeta_max = eBeta[0], eBeta[1]
        self.rBeta_min, self.rBeta_max = rBeta[0], rBeta[1]
    
    def __rescale(self, channel):
        # channelMin, channelMax = 0.0, 1.0
        channelMin, channelMax = np.amin(channel), np.amax(channel)
        if channelMin != channelMax:
            rescaledChannel = ((channel-channelMin)/(channelMax-channelMin)*255).astype(int)
        else:
            rescaledChannel = (channel*255).astype(int)
        return rescaledChannel
    
    def _augment_images(self, images, random_state, parents, hooks):
        numImages = len(images)
        hAlphaValues, hBetaValues = np.random.uniform(self.hAlpha_min, self.hAlpha_max, numImages), np.random.uniform(self.hBeta_min, self.hBeta_max, numImages)
        eAlphaValues, eBetaValues = np.random.uniform(self.eAlpha_min, self.eAlpha_max, numImages), np.random.uniform(self.eBeta_min, self.eBeta_max, numImages)
        rAlphaValues, rBetaValues = np.random.uniform(self.rAlpha_min, self.rAlpha_max, numImages), np.random.uniform(self.rBeta_min, self.rBeta_max, numImages)
        
        for image, hAlpha, eAlpha, rAlpha, hBeta, eBeta, rBeta in zip(images, hAlphaValues, eAlphaValues, rAlphaValues, hBetaValues, eBetaValues, rBetaValues):

            hedImage = rgb2hed(image)
            
            h, e, r = hedImage[:, :, 0], hedImage[:, :, 1], hedImage[:, :, 2]
            h, e, r = h*hAlpha + hBeta, e*eAlpha + eBeta, r*rAlpha + rBeta
            newHedImage = np.dstack((h, e, r))
            newRgbImage = hed2rgb(newHedImage)
            
            newRgbImageRescaled = np.empty_like(image)
            newRgbImageRescaled[:, :, 0] = self.__rescale(newRgbImage[:, :, 0])
            newRgbImageRescaled[:, :, 1] = self.__rescale(newRgbImage[:, :, 1])
            newRgbImageRescaled[:, :, 2] = self.__rescale(newRgbImage[:, :, 2])
            image[...] = newRgbImageRescaled
            
        return images
    
    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        pass
    
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        pass
    
    def get_parameters(self):
        return [(self.hAlpha_min, self.hAlpha_max), (self.eAlpha_min, self.eAlpha_max), (self.rAlpha_min, self.rAlpha_max), (self.eBeta_min, self.eBeta_max), (self.eBeta_min, self.eBeta_max), (self.hBeta_min, self.hBeta_max)]
