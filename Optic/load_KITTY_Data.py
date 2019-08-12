import numpy as np
import KITTY_Data as KITTY_Data
import h5py
import yaml
import pdb

config_file = "/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/configfile.yaml"
filenames_file = "/users/start2012/r0298867/Downloads/KITTY/kitti_train_files.txt"
# data_path = "/users/start2012/r0298867/Downloads/KITTY/"
# data_path = "/esat/citrine/troussel/IROS19_Depth_Estim/KITTI/raw/"
data_path = "/esat/opal/r0298867/r0298867/datasets/kitti_raw/"

def readconfigFile(filename):
    with open(filename, 'r') as cf:
        conf = yaml.load(cf.read())
        return conf

def readdataFile(filename):
    with open(filename, 'r') as cf:
        pathfile = cf.readlines()
        return pathfile

def prepare_data(rootDataFolder, data_file, config_File):
    return KITTY_Data.KITTY_Data(rootDataFolder, data_file, config_File)

def main():
    # Prepare training data
    print "Preparing data - KITTY"
    conf = readconfigFile(config_file)
    data_path_file = readdataFile(filenames_file)

    dataGenerator = prepare_data(data_path, data_path_file, conf)
    dataGenerator.__getitem__(1)

    return dataGenerator, conf
	


	
if __name__ == '__main__':
    main()