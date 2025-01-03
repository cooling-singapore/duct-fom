def get_getis_ord_global(D,W,xbar,S,n):
    import numpy as np



    ncols, nrows = D.shape

    Xbar = xbar
    temp=np.multiply((D-Xbar), W)
    num=np.nansum(temp)

    temp1=np.multiply(W, W)
    temp1=np.nansum(temp1)
    temp2=np.nansum(W)
    den = S*np.sqrt(1/(n-1))*np.sqrt(n*temp1-temp2**2)
    G=num/den
    return G