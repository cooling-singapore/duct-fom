import matplotlib.pyplot as plt
import numpy as np

from ExtractFeatures import ExtractFeatures
from SpatialHotSpotAnalysis import SpatialHotSpotAnalysis
from ReadData import ReadData
from LCP_analysis import LCP_analysis

###########################################################
TypeOfAnalysis = 1

if TypeOfAnalysis ==0:
    UnitSize = 10
    SpatialNeighborhoodSize = 9
    r=np.remainder(SpatialNeighborhoodSize,2)
    if r==0:
        print('SpatialNeighborhoodSize should be an odd number')
        exit()
else:
    UnitSize = 30
    DirectionWind=1
#####################################################
# read mask and shape files
#####################################################
InputFile1 = 'C:/Users/ido/Dropbox/CoolingSingapore/CS 2/Projects/Mesoscale Design guidelines/Wind corridors/Matlab/Code/MaskInput.tiff'
InputFile2 = 'C:/Users/ido/Dropbox/CoolingSingapore/CS 2/Projects/Mesoscale Design guidelines/Wind corridors/Matlab/Code/building-footprints.tiff'


#InputFile1 = 'C:/Users/ido/Dropbox/CoolingSingapore/CS 2/Projects/Mesoscale Design guidelines/Wind corridors/Matlab/Code/MaskAdel.tiff'
#InputFile2 = 'C:/Users/ido/Dropbox/CoolingSingapore/CS 2/Projects/Mesoscale Design guidelines/Wind corridors/Matlab/Code/DUCT_Tengah_S05_IN.tiff'

bh = ReadData(InputFile1,InputFile2)
FAD,Lambda_p, VABH,Mask, FAD_mean, z_d,z_0 = ExtractFeatures(bh,UnitSize)
###########################################################

###########################################################
if TypeOfAnalysis ==0:
    #####################################################
    # Run spatial analysis
    #####################################################
    GetisORd_statistic = SpatialHotSpotAnalysis(FAD_mean, SpatialNeighborhoodSize, Mask)
else:
    print('Executing 1 iteration of LCP out of 4')
    BetweenessMetric0 = LCP_analysis(FAD,Mask, 0)
    print('Executing 2 iteration of LCP out of 4')
    BetweenessMetric1 = LCP_analysis(FAD, Mask,1)
    print('Executing 3 iteration of LCP out of 4')
    BetweenessMetric2 = LCP_analysis(FAD, Mask,2)
    print('Executing 4 iteration of LCP out of 4')
    BetweenessMetric3 = LCP_analysis(FAD, Mask,3)
    BetweenessMetric  = BetweenessMetric0+BetweenessMetric1+BetweenessMetric2+BetweenessMetric3
#################################################################
#################################################################
#################################################################
#################################################################
######################################################
# plot results
#####################################################
T=0.1
if TypeOfAnalysis ==0:
    fig, axs = plt.subplots(1)
    fig.suptitle('Hot-Cold spots')
    plt.imshow(GetisORd_statistic**2, cmap='jet', interpolation='nearest')
    plt.imshow(Mask, alpha=T)
    #plt.show()
    plt.savefig('GO_Adel_OSM_mod_S03.png')
else:
    # create figure
    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = 2


    fig.add_subplot(rows, columns, 1)
    plt.imshow(BetweenessMetric0)
    plt.axis('off')
    plt.title('North-South direction')
    plt.imshow(Mask, alpha=T)


    fig.add_subplot(rows, columns, 2)
    plt.imshow(BetweenessMetric1)
    plt.axis('off')
    plt.title('East-West direction')
    plt.imshow(Mask, alpha=T)

    fig.add_subplot(rows, columns, 3)
    plt.imshow(BetweenessMetric2)
    plt.axis('off')
    plt.title('North-West South-East direction')
    plt.imshow(Mask, alpha=T)

    fig.add_subplot(rows, columns, 4)
    plt.imshow(BetweenessMetric3)
    plt.axis('off')
    plt.title('North-East South-West direction')
    plt.imshow(Mask, alpha=T)

    fig.add_subplot(rows, columns, 5)
    plt.imshow(BetweenessMetric)
    plt.axis('off')
    plt.title('Overall directions')
    plt.imshow(Mask,alpha=T)


    fig.add_subplot(rows, columns, 6)
    plt.imshow(FAD_mean)
    plt.axis('off')
    plt.title('Average FAD')
    plt.imshow(Mask,alpha=T)
    plt.show()


    plt.savefig('DUCT.png')
    #plt.savefig('DUCT_Tengah_100_S05.png')

x=1