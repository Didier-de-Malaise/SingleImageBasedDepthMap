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
        @config: python dictionary containing parameters about the data.
        @rootDataFolder: root folder containing the RGB data
        @hdfFn: Path to the .hdf5 file
    """

    def __init__(self, rootDataFolder, hdfFn, config):
        self.rootDataFolder = rootDataFolder
        self.hdfFile = h5py.File(hdfFn, 'r')
        self.config = config
        self.batchAm = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0] / config["batchSize"]))

        print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))

    def __getitem__(self, key):
        if key >= self.batchAm:
            raise IndexError
        else:
            batch = range(key * self.config["batchSize"], (key + 1) * self.config["batchSize"])
            # print("Loading batch %d" % key)

            rgb, gtDepth = self.load_batch(batch)
            # print("Batch loaded")
            return rgb, gtDepth

    def __len__(self):
        return self.batchAm

    def load_batch(self, batch):
        rgb = np.ones((self.config["batchSize"], self.config["H"], self.config["W"], 3))
        depth = np.ones((self.config["batchSize"], self.config["HOut"], self.config["WOut"]))

        i = 0
        start1 = time()
        for entry in batch:
            # Get rgb file
            a = time()
            num = self.hdfFile["depth"]["depth_folder_id"][int(entry)]
            b = time()
            name = self.hdfFile["depth"]["depth_labels"][int(entry)]
            c = time()
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
            # rgb[i,:,:,:] = scipy.misc.imread(match, mode = "RGB")
            start4 = end3 = time()
            # Get GT depth info
            start5 = end4 = time()
            # print(b-a)
            # print(c-b)
            # print(end4-c)

            end4 = time()
            i += 1
        # print("Reading ID: %f, reading label: %f, reading image: %f, reading depth: %f" % (end1-start1,end2-start2,end3-start3,end4-start4))
        # print "rgb matrix shape = ", rgb.shape
        start2 = end1 = time()
        depth[:, :, :] = np.swapaxes(
            np.swapaxes(self.hdfFile["depth"]["depth_data"][:, :, sorted([int(x) for x in batch])], 0, 2), 1, 2)
        end2 = time()
        # print("Reading RGB data: %f reading depth: %f" % (end1-start1,end2-start2))
        return rgb, depth











# import numpy as np
# from time import time
# import scipy.misc
# from math import floor
# import PIL.Image as im
# import h5py
# import pdb
# import random


# # TODO: Replace config parameter and replace with batchSize. Other parameters can be taken from hdf file, makes it easier and more generic
# class NYU_Data(object):
#     """
#         Object that handles loading NYU data from an HDF file
#         Is iterable
#         @config: python dictionary containing parameters about the data.
#         @rootDataFolder: root folder containing the RGB data
#         @hdfFn: Path to the .hdf5 file
#     """

#     def __init__(self, rootDataFolder, hdfFn, config):
#         self.rootDataFolder = rootDataFolder
#         self.hdfFile = h5py.File(hdfFn, 'r')
#         self.config = config
#         self.batchAm = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0] / config["batchSize"]))
#         # random_list = random.sample(list, len(list))
#         self.numberImages = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0]))
#         self.list = [i for i in range(self.numberImages)]
#         self.randomList = random.sample(self.list, len(self.list))
#         self.numlist = []

#         print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))
#         # for i in range(1,10):
#         #     print self.randomList[i]

#     def shuffle(self, number_img_val, num_img_train):
#         self.randomList = self.randomList[:number_img_val] + random.sample(self.randomList[number_img_val:num_img_train], len(self.randomList[number_img_val:num_img_train])) + self.randomList[num_img_train:]
#         # start = time()
#         # for x in self.list:
#         #     self.numlist.append(self.hdfFile["depth"]["depth_folder_id"][int(self.randomList[x])])
#         # end = time()

#         # print(end - start)

#     def __getitem__(self, key):
#         if key >= self.batchAm:
#             raise IndexError
#         else:
#             batch = range(key * self.config["batchSize"], (key + 1) * self.config["batchSize"])
#             # print("Loading batch %d" % key)

#             rgb, gtDepth = self.load_batch(batch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
#             # print("Batch loaded")
#             return rgb, gtDepth

#     def __len__(self):
#         return self.batchAm

#     def load_batch(self, batch):
#         rgb = np.ones((self.config["batchSize"], self.config["H"], self.config["W"], 3))
#         depth = np.ones((self.config["batchSize"], self.config["HOut"], self.config["WOut"]))

#         i = 0
#         start1 = time()
#         for entry in batch:
#             # Get rgb file
#             a = time()
#             num = self.hdfFile["depth"]["depth_folder_id"][int(self.randomList[entry])]
#             b = time()
#             # num = self.numlist[entry]

#             name = self.hdfFile["depth"]["depth_labels"][int(self.randomList[entry])]
#             c= time()
#             match = "%s/imgs_%d/%s.jpeg" % (self.rootDataFolder, num, name)
#             # print match
#             f = open(match, 'rb')
#             # resize images to 228*304
#             pilIM = im.open(f)
#             new_width = 304
#             wpercent = new_width / float(pilIM.size[0])
#             new_height = int((float(pilIM.size[1]) * float(wpercent)))
#             pilIM = pilIM.resize((new_width, new_height))

#             pilIm2 = pilIM.copy()  # PIL bug workaround
#             f.close()
#             rgb[i, :, :, :] = np.asarray(pilIM)
#             pilIM.close()
#             # rgb[i,:,:,:] = scipy.misc.imread(match, mode = "RGB")
#             start4 = end3 = time()
#             # Get GT depth info
#             start5 = end4 = time()
#             end4 = time()
#             depth[i, :, :] = self.hdfFile["depth"]["depth_data"][:, :, int(self.randomList[entry])  ]
#             d = time()
#             # print(b-a)
#             # print(c-b)
#             # print(end4-c)
#             # print(d-end4)
#             # depth[i, :, :] = np.swapaxes(np.swapaxes(self.hdfFile["depth"]["depth_data"][:, :, int(self.randomList[entry]) ], 0, 2), 1, 2)

#             i += 1
#         # print("Reading ID: %f, reading label: %f, reading image: %f, reading depth: %f" % (end1-start1,end2-start2,end3-start3,end4-start4))
#         # print "rgb matrix shape = ", rgb.shape
#         start2 = end1 = time()
#         print(end1-start1)
#         pdb.set_trace()
#         # depth[:, :, :] = np.swapaxes(
#         #     np.swapaxes(self.hdfFile["depth"]["depth_data"][:, :, sorted([int(self.randomList[x]) for x in batch])], 0, 2), 1, 2)
#         end2 = time()
#         # print(end2-start1)
#         # print("Reading RGB data: %f reading depth: %f" % (end1-start1,end2-start2))
#         return rgb, depth
