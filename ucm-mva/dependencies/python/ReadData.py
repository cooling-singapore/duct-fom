def ReadData(InputFile1,InputFile2):
    import matplotlib.pyplot as plt
    import numpy as np

    #####################################################
    # read mask and shape files
    #####################################################
    LandMaskSG = plt.imread(InputFile1)
    BinaryMapSG = np.sign(LandMaskSG)
    temp = plt.imread(InputFile2)

    ###################
    # if temp is a read only array than the np.copy opens it
    ###################
    bh = np.copy(temp)
    #####################################################
    # clean the outliers and adjust values
    #####################################################
    water = np.where(BinaryMapSG == 0)
    bh[water] = 255
    Outliers = np.where(bh > 1000)
    bh[Outliers] = 255



    return bh