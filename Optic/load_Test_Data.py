from time import time
from math import floor
import Test_Data as Test_Data
import PIL.Image as im
import tensorflow as tf
import numpy as np
import scipy.misc
import h5py
import yaml
import pdb

config_file = "/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/configfile.yaml"
data_path = "/esat/opal/r0298867/r0298867/datasets/kitti_optic/data_scene_flow/training/"

def readconfigFile(filename):
    with open(filename, 'r') as cf:
        conf = yaml.load(cf.read())
        return conf

def prepare_data(rootDataFolder, config_File):
    return Test_Data.Test_Data(rootDataFolder, config_File)

def main():
    print "Preparing data - Test optic (KITTI)"
    conf = readconfigFile(config_file)
    dataGenerator = prepare_data(data_path, conf)
    # dataGenerator.__getitem__(1)
    return dataGenerator, conf
	
if __name__ == '__main__':
    main()
