def SpatialHotSpotAnalysis(F,N,Mask):
    import numpy as np
    from Getis_Ord_V1 import Getis_Ord_V1

    W = np.ones((N, N))
    temp=np.ceil(N/2)
    wsx = int(temp)
    wsy = int(temp)
    W[wsx-1, wsy-1] = 0
    W = W / (N ** 2 - 1)

    GO_statistc = Getis_Ord_V1(F, W, Mask)

    return GO_statistc