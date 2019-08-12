import numpy as np
import NYU_Data as NYU_Data
import h5py
import yaml
import pdb

# hdf5_file_name = '/esat/qayd/tmp/NYU_lzif.hdf5'    #Dieptebeelden en labels
# hdf5_file_name = '/users/start2012/r0298867/Thesis/implementation1/test_code/hdf_file/testh5py.hdf5'

'''
# De gemengde hdf5 file gaat tot 460000 afbeeldingen
# De originele hdf5 file gaat tot 462032 dus er zijn 2032 testafbeeldingen
# Voor de testdata verander hier hdf5_file_name en uncomment feed_dict/ __getitem__() in coarse.py
'''


hdf5_file_name = '/esat/opal/r0298867/r0298867/testh5py.hdf5' #Dieptebeelden en labels
# hdf5_file_name = '/esat/opal/r0298867/r0298867/tmp/NYU_lzif.hdf5' #Dieptebeelden en labels
#rootData = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/" #RGB images
# rootData = "/esat/citrine/troussel/IROS19_Depth_Estim/Stijn_Data/NYUv2/processed"
rootData = '/esat/opal/r0298867/r0298867/datasets/Stijn_Data/NYUv2/processed'

configFile = "/users/start2012/r0298867/Thesis/implementation1/build_new/Coarse/configfile.yaml"

def readconfigFile(filename):
    with open(filename, 'r') as cf:
        conf = yaml.load(cf.read())
        return conf

def prepare_data(hdf5_fn, rootDataFolder, config_File):
    return NYU_Data.NYU_Data(rootDataFolder, hdf5_fn, config_File)

def main():
    # Prepare training data
    print "Preparing data"
    conf = readconfigFile(configFile)
    dataGenerator = prepare_data(hdf5_file_name, rootData, conf)

    return dataGenerator, conf

if __name__ == '__main__':
    main()