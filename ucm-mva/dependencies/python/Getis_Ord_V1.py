def Getis_Ord_V1(grid, W, Mask):
    import numpy as np
    from get_getis_ord_global import get_getis_ord_global

    a1,b1=grid.shape
    n=a1*b1
    ###########################
    A = grid * Mask
    temp1 = np.reshape(Mask, (a1 * b1, 1))
    index = np.where(temp1> 0)
    N = len(index[1])
    xbar = np.sum(A) / N
    temp = np.reshape(A, (a1 * b1, 1))
    temp1 = np.dot(temp[index],temp[index])
    S = np.sqrt(temp1 / N - xbar ** 2)
   ###########################
    ## Getis   Ord     calculations
    M = np.zeros((a1,b1))
    a,b=W.shape

    wsx = int(np.floor(a / 2))
    wsy = int(np.floor(b/ 2))
######################################
    Start=1+wsy-1
    End=a1-wsy-1
    for row in range(Start,End):
        print('{:,.2f}  GO analysis completed'.format(row/(End-Start)*100))
        for col in range(1+wsx-1, b1-wsx):
#            print('{:,.2f}  col completed'.format(col))

            D=grid[row-wsx:row+wsx+1:1,col-wsy:col+wsy+1:1]
            M[row,col] = get_getis_ord_global(D,W,xbar,S,n)

    ######################
    # offset to 0
    ######################
    m=np.min(M)
    M = M-m
    return M