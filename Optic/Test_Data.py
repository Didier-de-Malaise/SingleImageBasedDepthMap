from time import time
from math import floor
import PIL.Image as im
import tensorflow as tf
import numpy as np
import scipy.misc
import h5py
import yaml
import pdb


class Test_Data(object):
    """
        Object that handles loading KITTY ground truth optical flow / disparity data
        @config: python dictionary containing parameters about the data.
        @rootDataFolder: root folder containing the RGB data
    """

    def __init__(self, rootDataFolder, config):
        self.rootDataFolder = rootDataFolder
        self.config = config

        # This line calculates the amount of batches. There are 200 training and 200 test image pairs with groundtruth.
        # Test/training images is only the name in that dataset, every image pair can be used as test data in my network.
        # In this case I use 200 trainings image pairs folder
        self.batchAm = int(floor(200 / (config["batchSize"])))
        print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))

    def __getitem__(self, key):
        # Load batch, check if index is out of bounds
        if key >= self.batchAm:
            print ('key: %d' % key)
            raise IndexError
        else:
            batch = range(key * self.config["batchSize"], (key + 1) * self.config["batchSize"])
            rgb1, rgb2, gt = self.load_batch(key, batch)
            return rgb1, rgb2, gt

            
    def load_batch(self, key, batch):
        flow = True

        rgb_1 = np.ones((self.config["batchSize"], 224, 224, 3))
        rgb_2 = np.ones((self.config["batchSize"], 224, 224, 3))
        if flow == False:
            gt = np.ones((self.config["batchSize"], 224, 224, 1))
        else:
            gt = np.ones((self.config["batchSize"], 224, 224, 3))

        i = 0
        for entry in batch:
            image_name = str(entry).zfill(6)

            string1 = 'image_2/' + image_name + '_10.png'
            if flow == False:
                string2 = 'image_3/' + image_name + '_10.png' # Voor disparity
                string3 = 'disp_occ_0/' + image_name + '_10.png' # ground truth voor disparity
            else:
                string2 = 'image_2/' + image_name + '_11.png' # Voor optical flow
                string3 = 'flow_occ/' + image_name + '_10.png' # ground truth voor optical flow

            f1 = open(self.rootDataFolder + string1, 'rb')
            f2 = open(self.rootDataFolder + string2, 'rb')
            f3 = open(self.rootDataFolder + string3, 'rb')

            # resize images to 224*224
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
            

            pilIM = im.open(f3)
            new_height = 224
            wpercent = new_height / float(pilIM.size[1])
            new_width = int((float(pilIM.size[0]) * float(wpercent)))
            pilIM = pilIM.resize((new_width, new_height))
            pilIM = pilIM.crop((0,0,224,224))
            pilIm2 = pilIM.copy()
            f3.close()
            if flow == False:
                gt[i, :, :, 0] = np.asarray(pilIM)
            else:
                gt[i, :, :, :] = np.asarray(pilIM)
            pilIM.close()

            i += 1
            
        return rgb_1, rgb_2, gt
