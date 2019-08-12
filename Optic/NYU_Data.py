import numpy as np
from time import time
import scipy.misc
from math import floor
import PIL.Image as im
import h5py
import pdb



# TODO: Replace config parameter and replace with batchSize. Other parameters can be taken from hdf file, makes it easier and more generic
class NYU_Data(object):
    """
        Object that handles loading NYU data from an HDF file
        Is iterable
        @rootDataFolder: root folder containing the RGB data
        @pathfile: path to the images starting from rootDataFolder
        @config: python dictionary containing parameters about the data.
    """

    def __init__(self, rootDataFolder, hdfFile, config):
        self.rootDataFolder = rootDataFolder
        self.hdfFile = h5py.File(hdfFile, 'r')
        self.config = config
        self.batchAm = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0] / config["batchSize"]))

        print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))

    def __getitem__(self, key):
        if key >= self.batchAm:
            raise IndexError
        else:
            batch = range(key * self.config["batchSize"], (key + 1) * self.config["batchSize"])
            # print("Loading batch %d" % key)

            rgb = self.load_batch(batch)
            # rgb, gtDepth = self.load_batch(batch)
            # print("Batch loaded")
            return rgb #, gtDepth

    def __len__(self):
        return self.batchAm

    def load_batch(self, batch):
        rgb = np.ones((self.config["batchSize"], self.config["H"], self.config["W"], 3))
        depth = np.ones((self.config["batchSize"], self.config["HOut"], self.config["WOut"]))

        i = 0

        for entry in batch:
            num = self.hdfFile["depth"]["depth_folder_id"][int(entry)]
            name = self.hdfFile["depth"]["depth_labels"][int(entry)]
            match = "%s/imgs_%d/%s.jpeg" % (self.rootDataFolder, num, name)
            f = open(match, 'rb')
            # resize images to 228*304
            pilIM = im.open(f)
            new_width = 304
            wpercent = new_width / float(pilIM.size[0])
            new_height = int((float(pilIM.size[1]) * float(wpercent)))
            pilIM = pilIM.resize((new_width, new_height))

            pilIm2 = pilIM.copy()  # PIL bug workaround
            f.close()
            rgb[i, :, :, :] = np.asarray(pilIM)
            pilIM.close()
            i += 1

        return rgb#, depth






