import numpy as np
import NYU_Data as NYU_Data
import PIL.Image as im
import tensorflow as tf
from time import time
from math import floor
import h5py
import yaml
import pdb
import scipy.misc

class KITTY_Data(object):
    """
        Object that handles loading KITTY data
        @config: python dictionary containing parameters about the data.
        @rootDataFolder: root folder containing the RGB data
    """

    def __init__(self, rootDataFolder, pathfile, config):
        self.rootDataFolder = rootDataFolder
        self.pathfile = pathfile
        self.config = config
        self.batchAm = int(floor(len(pathfile) / (config["batchSize"])))
        print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))

    def __getitem__(self, key):
        # Load batch, check if index is out of bounds
        if key >= self.batchAm:
            print ('key: %d' % key)
            raise IndexError
        else:
            batch = range(key * self.config["batchSize"], (key + 1) * self.config["batchSize"])
            rgb1, rgb2 = self.load_batch(batch)
            return rgb1, rgb2

            
    def load_batch(self, batch):
        rgb_1 = np.ones((self.config["batchSize"], 224, 224, 3))
        rgb_2 = np.ones((self.config["batchSize"], 224, 224, 3))

        i = 0
        for entry in batch:
            string1 = self.pathfile[entry].split(" ")[0]
            string2 = self.pathfile[entry].split(" ")[1].rstrip()

            f1 = open(self.rootDataFolder + string1, 'rb')
            f2 = open(self.rootDataFolder + string2, 'rb')

            pilIM = im.open(f1)
            new_height = 224
            wpercent = new_height / float(pilIM.size[1])
            new_width = int((float(pilIM.size[0]) * float(wpercent)))
            pilIM = pilIM.resize((new_width, new_height))
            pilIM = pilIM.crop((0,0,224,224))
            pilIm2 = pilIM.copy()

            f1.close()

            rgb_1[i, :, :, :] = np.asarray(pilIM)
            pilIM.close()

            pilIM = im.open(f2)
            new_height = 224
            wpercent = new_height / float(pilIM.size[1])
            new_width = int((float(pilIM.size[0]) * float(wpercent)))
            pilIM = pilIM.resize((new_width, new_height))
            pilIM = pilIM.crop((0,0,224,224))
            pilIm2 = pilIM.copy()
            f2.close()

            rgb_2[i, :, :, :] = np.asarray(pilIM)
            pilIM.close()

            i += 1
            
        return rgb_1, rgb_2
