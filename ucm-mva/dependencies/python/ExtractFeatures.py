def ExtractFeatures(Data,s):
    import numpy as np
    from FAD_calc_V1 import FAD_calc_V1
    from Volumetric_averaged_building_height import Volumetric_averaged_building_height
    import math

    THETA = np.arange(0, 360 - 5, 360 / 8)
    N = np.shape(THETA)

    a, b = np.shape(Data)
    a = math.floor(a / s)
    b = math.floor(b / s)


    WaterParameter = 255
    Mask=np.zeros((a,b))
    FAD = np.array(np.zeros((a,b,N[0])))
    Lambda_p = np.array(np.zeros((a, b)))
    VABH = np.array(np.zeros((a, b)))
    for itr1 in range(0, a):

        print('{:,.2f} % of Feature Extraction completed'.format(itr1/a*100))
        x = np.arange(itr1 * s ,(itr1+1) * s+1,1)
        x_max = np.max(x)
        x_min = np.min(x)

        for itr2 in range(0,b):
            y = np.arange(itr2  * s ,(itr2+1) * s+1,1)
            y_max = np.max(y)
            y_min = np.min(y)

            ImageSubSample = Data[x_min:x_max:1, y_min:y_max:1]

            Max = np.max(ImageSubSample)
            Min = np.min(ImageSubSample)
            Mask[itr1, itr2] = 1
            if Min == WaterParameter:
                FAD[itr1, itr2, :]=0
                Lambda_p[itr1, itr2] = 0
                VABH[itr1, itr2] = 0
                Mask[itr1, itr2] = 0
            elif Max == 0:
                FAD[itr1, itr2, :]=0
                Lambda_p[itr1, itr2] = 0
                VABH[itr1, itr2] = 0
            elif Max == WaterParameter and  Min < WaterParameter:
                ImageSubSample_v = np.reshape(ImageSubSample, (s * s, 1))
                indx = np.where(ImageSubSample_v== WaterParameter)
                ImageSubSample_v[indx] = 1
                ImageSubSample = np.reshape(ImageSubSample_v, (s, s))

                #FAD[itr1, itr2,np.arrange(1,N,1)], Lambda_p[itr1, itr2]=FAD_calc_V1(ImageSubSample, THETA)
                temp1, temp2 = FAD_calc_V1(ImageSubSample, THETA)
                FAD[itr1, itr2, :] = temp1
                Lambda_p[itr1, itr2] = temp2

                VABH[itr1, itr2] = Volumetric_averaged_building_height(ImageSubSample)
            else:
                temp1 ,temp2=FAD_calc_V1(ImageSubSample, THETA)
                FAD[itr1, itr2, :]=temp1
                Lambda_p[itr1, itr2]=temp2

                VABH[itr1, itr2] = Volumetric_averaged_building_height(ImageSubSample)

##################################################
    kappa = 0.4
    eps = 10 ** (-8)
    FAD_mean = np.mean(FAD, axis=2) + eps
    temp = np.power(Lambda_p, 0.6)
    z_d = np.multiply(VABH, temp)
    temp = VABH - z_d
    temp2 = np.power(FAD_mean, -1)
    temp3 = np.sqrt(temp2)
    temp1 = np.exp(-temp3)
    z_0 = np.multiply(temp, temp1)
##################################################
    return FAD,Lambda_p, VABH,Mask,FAD_mean,z_d,z_0