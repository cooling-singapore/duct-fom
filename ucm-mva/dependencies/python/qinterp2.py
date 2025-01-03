def qinterp2(X,Y,Z,xi,yi):
    import numpy as np

    ImageSize_x,ImageSize_y =  np.shape(X)
    ndx = 1/(X[1,2]-X[1,1])
    ndy = 1/(Y[2,1]-Y[1,1])
    xi = (xi - X[1,1])*ndx
    yi = (yi - Y[1,1])*ndy
    Zi = float('nan')*np.ones(np.shape(xi))
    rxi = np.round(xi)+2
    ryi = np.round(yi)+2

    a, b = np.shape(Z)
    Zi_v = np.reshape(Zi,(1,a*b))

    temp1=np.reshape(1*(rxi>0),(1,a*b))
    temp2 = np.reshape(1*(rxi<=ImageSize_y),(1,a*b))
    #temp3 = not math.isnan(rxi)
    temp4 = np.reshape(1*(ryi>0),(1,a*b))
    temp5 = np.reshape(1*(ryi<=ImageSize_x),(1,a*b))
    #temp6 = ~isnan(ryi)
    index1 = np.multiply(temp1, temp2)
    index2 = np.multiply(temp4, temp5)
    index= np.nonzero(np.multiply(index1,index2))
    index3=index[1]

    ind = ryi + ImageSize_x*(rxi-1)
    ind_v = np.reshape(ind, (1, a * b))
    temp =ind_v[0][index3]-1
    temp1=temp.astype(int)

    Z_v=np.reshape(Z,(1,a*b))
    Zi_v[0][index3] = Z_v[0][temp1]
    Zi=np.reshape(Zi_v,(a,b))
    return Zi



