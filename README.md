# SatelliteImageClassification
CSC 422/522 project

This is mostly a Python project that uses numpy, scipy, sift, and sklearn. There are two different stages to running the model.

The first step is to generate the sift and pca output vectors from the training and testing images. This can be accomplished by running tests/SiftDataGenerator.py. To run the data generator, sift must be on your path and has to be accessible from the command line. Running it is not necessary unless the training or testing sets have been changed as the .dat files it produces are already in the repository. If running on windows, you may have to run data/fixdataforwindows.py to put them in the correct format for un-pickling.

After the data generator is run, both the K-nearest-neighbor and artificial neural network tests can be run from the files data/SiftKNN1Test.py and tests/SiftANNTest.py  These files train and test the models based on the data files generated in the first step.
