import numpy as np
def non_mamima_suppression(m, window=10, thresh=0.2):
    '''
    m: binary mask generated by network
    window: size of tiles to limit maximum
    thresh: min value of point to be considered
    '''
    #3D->2D
    m = m.reshape(m.shape[1::-1])
    new_m = np.zeros(m.shape)
    #iterate over window x window sized tiles
    for ii in range(0,m.shape[0],window):
        for jj in range(0,m.shape[1],window):
            w = m[ii:ii+window,jj:jj+window]
            # pick max value in window
            w_max_idx = np.unravel_index(w.argmax(), w.shape)
            if w[w_max_idx] > thresh:
                new_m[w_max_idx[0]+ii, w_max_idx[1]+jj] = 1
    #2D->3D
    return new_m.reshape((*m.shape[::-1],1))