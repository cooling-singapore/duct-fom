def LCP_analysis(FAD, Mask, DirectionWind):
    import numpy as np
    import igraph as igraph
    import numpy.matlib
    import scipy.ndimage

    import warnings

    #warnings.filterwarnings("error")

    warnings.filterwarnings("ignore")

    #########################################################

    def yx_to_idx(yx: (int, int), shape: (int, int)) -> int:
        return yx[0] * shape[1] + yx[1]

    #########################################################
    def idx_to_yx(idx: int, shape: (int, int)) -> (int, int):
        y = int(idx / shape[1])
        x = int(idx % shape[1])
        return y, x

    #########################################################
    def add_weighted_edge(origin: (int, int), delta: (int, int), shape: (int, int), fad: np.ndarray, edges: list,
                          weights: list, Delta) -> None:
        destination = [origin[0] + delta[0], origin[1] + delta[1]]

        # is the destination valid, i.e., on the grid?
        if 0 <= destination[0] < shape[0] and 0 <= destination[1] < shape[1]:
            idx0 = yx_to_idx(origin, shape)
            idx1 = yx_to_idx(destination, shape)
            edges.append((idx0, idx1))
            weights.append(fad[origin[0]][origin[1]] + Delta)

    #########################################################
    def create_graph(fad: np.ndarray, mask: np.ndarray, Penalty: np.ndarray):
        shape = (fad.shape[1], fad.shape[2])
        edges = []
        weights = []
        for iy in range(shape[0]):
            for ix in range(shape[1]):
                # ignore masked cells
                if mask[iy][ix] == 0:
                    continue

                # vertical edges
                add_weighted_edge((iy, ix), (-1, 0), shape, fad[0], edges, weights, Penalty[0])  # N->S (down)
                add_weighted_edge((iy, ix), (+1, 0), shape, fad[4], edges, weights, Penalty[1])  # S->N (up)

                # horizontal edges
                add_weighted_edge((iy, ix), (0, -1), shape, fad[2], edges, weights, Penalty[2])  # E->W (left)
                add_weighted_edge((iy, ix), (0, +1), shape, fad[6], edges, weights, Penalty[3])  # W->E (right)

                # diagonal edges
                add_weighted_edge((iy, ix), (-1, -1), shape, fad[1], edges, weights, Penalty[4])  # NE->SW (down-left)
                add_weighted_edge((iy, ix), (-1, +1), shape, fad[5], edges, weights, Penalty[5])  # NW->SE (down-right)
                add_weighted_edge((iy, ix), (+1, -1), shape, fad[3], edges, weights, Penalty[6])  # SE->NW (down-left)
                add_weighted_edge((iy, ix), (+1, +1), shape, fad[7], edges, weights, Penalty[7])  # SW->NE (down-right)

        # create directed graph
        g = igraph.Graph(edges, directed=True)
        g.es['weight'] = weights

        return g

    #########################################################
    # def determine_vertical_od_pairs(mask: np.ndarray, inverse: bool) -> list[((int, int), (int, int))]:
    def determine_vertical_od_pairs(mask, inverse):
        shape = mask.shape
        pairs0 = []
        for x in range(mask.shape[1]):
            idx0 = None
            y_range = range(mask.shape[0])
            y_range = reversed(y_range) if inverse else y_range
            for y in y_range:
                if idx0 is None and mask[y][x] == 1:
                    idx0 = yx_to_idx((y, x), shape)

                elif idx0 is not None and mask[y][x] == 0:
                    idx1 = yx_to_idx((y - 1, x), shape)

                    if idx0 != idx1:
                        if idx0 > 0 and idx1 > 0:
                            pairs0.append((idx0, idx1))
                        else:
                            x = 1

                    idx0 = None

        return pairs0

    #########################################################
    # def determine_horizontal_od_pairs(mask: np.ndarray, inverse: bool) -> list[((int, int), (int, int))]:
    def determine_horizontal_od_pairs(mask, inverse):
        shape = mask.shape
        pairs0 = []
        for y in range(mask.shape[0]):
            idx0 = None
            x_range = range(mask.shape[1])
            x_range = reversed(x_range) if inverse else x_range
            for x in x_range:
                if idx0 is None and mask[y][x] == 1:
                    idx0 = yx_to_idx((y, x), shape)

                elif idx0 is not None and mask[y][x] == 0:
                    idx1 = yx_to_idx((y, x - 1), shape)

                    if idx0 != idx1:
                        pairs0.append((idx0, idx1))

                    idx0 = None

        return pairs0

    #########################################################
    def determine_cone_od_pairs(od_pairs, cone_size):

        pairs = []
        O = []
        D = []
        L = len(od_pairs)
        pairs0 = []
        # for x in range(cone_size, L-cone_size,1):
        for x in range(0, L, 1):
            Start = max(0, x - cone_size)
            End = min(L, x + cone_size)
            ConeLength = End - Start
            # read source and duplicate it cone_size*2 times
            temp1 = od_pairs[x][0]
            temp2 = np.matlib.repmat(temp1, 1, ConeLength)
            #####################################
            temp3 = od_pairs[Start: End]

            for z in range(0, len(temp3), 1):
                D.append(temp3[z][1])
                O.append(temp2[0][1])

        pairs.append((O, D))
        pairs_OD = np.array(pairs).T.tolist()

        return pairs_OD

    #####################################################
    # set parameters for LCP analysis
    #####################################################
    cone_size = 4
    if DirectionWind == 0:
        od_pairs = determine_vertical_od_pairs(Mask, False)
        Penalty = [0, 10, 0.25, 0.25, 0, 0, 1, 1]
    elif DirectionWind == 1:
        od_pairs = determine_horizontal_od_pairs(Mask, True)
        Penalty = [2, 2, 0, 10, 0.5, 10, 0.5, 10]
    elif DirectionWind == 2:
        od_pairs = determine_vertical_od_pairs(Mask, False)
        x = []
        for a in range(len(od_pairs)):
            x.append(od_pairs[a][0])
        y = []
        for a in range(len(od_pairs)):
            y.append(od_pairs[a][1])

        a = len(x)
        z = int(np.round(a / 3))
        od_pairs = list(zip(x[0:2*z-10], y[a-2*z+10:a]))
        Penalty = [1, 0.5, 10, 0, 10, 0, 10, 100]
    elif DirectionWind == 3:
        od_pairs = determine_vertical_od_pairs(Mask, False)
        x = []
        for a in range(len(od_pairs)):
            x.append(od_pairs[a][0])
        y = []
        for a in range(len(od_pairs)):
            y.append(od_pairs[a][1])

        a = len(x)
        z = int(np.round(a / 3))
        od_pairs = list(zip(x[z:a], y[0:2*z]))
        Penalty = [0, 2, 0, 10, 0, 1, 0, 1]
    else:
        print('Incorrect direction chosen')
        return 0

    od_pairs_cone = determine_cone_od_pairs(od_pairs, cone_size)
    #####################################################
    FAD = np.transpose(FAD, (2, 0, 1))
    g = create_graph(FAD, Mask, Penalty)
    #####################################################
    # determine OD pairs
    #####################################################
    #####################################################
    # calculate paths from origin to destination for each OD pair
    #####################################################
    height = FAD.shape[1]
    width = FAD.shape[2]
    BetweenessMetric = np.zeros(shape=(height, width), dtype=np.uint16)
    #####################################################
    # LCP analysis
    #####################################################
    counter = 0
    a = len(od_pairs_cone)
    print('There are  {} LCPs to find'.format(a))
    for pair in od_pairs_cone:
       # print('Executing the  {} iteration of LCP out of {}'.format(counter, a))
        counter = counter + 1

        idx0 = int(np.round(pair[0]))
        idx1 = int(np.round(pair[1]))

        #if idx1!=2437:
        #    try:
        #        paths = g.get_shortest_paths(idx0, idx1, weights='weight')
        #    except RuntimeWarning:
        #        x=1
        #if idx1!=2437:
        paths = g.get_shortest_paths(idx0, idx1, weights='weight')

        for path in paths:
            for idx in path:
                iy, ix = idx_to_yx(idx, (height, width))
                BetweenessMetric[iy][ix] += 1

    #####################################################
    # plot results
    #####################################################
    #BetweenessMetric = BetweenessMetric * 10
    I = scipy.ndimage.gaussian_filter(BetweenessMetric, sigma=1.5, truncate=3)

    return I
