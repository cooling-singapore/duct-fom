def FAD_calc_V1(Image, THETA):
    import numpy as np
    from qinterp2 import qinterp2
    import math


    iLength, iWidth = np.shape(Image)
    iDiag = math.sqrt(iLength ** 2 + iWidth ** 2)
    LengthPad = math.ceil(iDiag - iLength) + 2
    WidthPad = math.ceil(iDiag - iWidth) + 2
    padImage = np.zeros((iLength + LengthPad, iWidth + WidthPad))
    temp1 = np.arange(math.ceil(LengthPad / 2), math.ceil(LengthPad / 2) + iLength , 1)
    temp1_min=np.min(temp1)
    temp1_max = np.max(temp1)+1
    temp2= np.arange (math.ceil(WidthPad / 2),  math.ceil(WidthPad / 2) + iWidth ,1)
    temp2_min = np.min(temp2)
    temp2_max = np.max(temp2)+1
    padImage[temp1_min:temp1_max,temp2_min:temp2_max] = Image


    n,t =  np.shape(padImage)
    x = np.linspace(-1, 1, n)
    X1, Y1 = np.meshgrid(x, x)
    eps = 10 ** (-8)
    n = len(THETA)
    FAD=np.zeros( n)

    for i in range(0,n):
        #print('Step 0 of 4: {:,.2f} % of XXXc completed'.format(i))

        FAR = np.zeros((t, n))

        theta = THETA[i] * math.pi / 180
        X = math.cos(theta) * X1 + -math.sin(theta) * Y1
        Y = math.sin(theta) * X1 + math.cos(theta) * Y1
        rot_image = qinterp2(X1,Y1,padImage,X,Y)

        a, b = np.shape(rot_image)
        rot_image_v = np.reshape(rot_image, (1, a * b))

        temp=np.array(rot_image_v)
        temp1=np.isnan(temp)
        rot_image_v[temp1] = 0
        M1 = np.max(padImage)
        M2 = np.max(rot_image_v)+eps
        rot_image_v = rot_image_v / M2 * M1

        rot_image = np.reshape(rot_image_v, (a, b))

        FAR = rot_image.max(0)
        temp=np.divide(np.sum(FAR) , (iLength * iWidth))
        FAD[i] = temp


    Lambda_p = np.divide(np.sum(rot_image > 0) , (iLength * iWidth))

    return FAD, Lambda_p