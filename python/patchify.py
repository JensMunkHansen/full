import numpy as np
import scipy
import collections
import random as rand


def get_indices_for_un_patchify(sImg,sP,step):
    ''' creates indices for fast patchifying and unpatchifying

    INPUTS:
      sx    image size
      sp    patch size
      step  offset between two patches (default == [1,1])

      OUTPUTS:
       patchInd             collection with indices
       patchInd.img2patch   patchifying indices
                            patch = img(patchInd.img2patch);
       patchInd.patch2img   unpatchifying indices

    NOTE: * for unpatchifying necessary to add a 0 column to the patch matrix
          * matrices are constructed row by row, as normally there are less rows than columns in the
            patchMtx
     '''
    lImg = np.prod(sImg)
    indImg = np.reshape(range(lImg), sImg)

    # no. of patches which fit into the image
    sB = (sImg - sP + step) / step

    lb              = np.prod(sB)
    lp              = np.prod(sP)
    indImg2Patch    = np.zeros([lp, lb])
    indPatch        = np.reshape(range(lp*lb), [lp, lb])

    indPatch2Img = np.ones([sImg[0],sImg[1],lp])*(lp*lb+1)

    # default value should be last column
    iRow   = 0;
    for jCol in range(sP[1]):
        for jRow in range(sP[0]):
            tmp1 = np.array(range(0, sImg[0]-sP[0]+1, step[0]))
            tmp2 = np.array(range(0, sImg[1]-sP[1]+1, step[1]))
            sel1                    = jRow  + tmp1
            sel2                    = jCol  + tmp2
            tmpIndImg2Patch = indImg[sel1,:]
            # do not know how to combine following 2 lines in python
            tmpIndImg2Patch = tmpIndImg2Patch[:,sel2]
            indImg2Patch[iRow, :]   = tmpIndImg2Patch.flatten()

            # next line not nice, but do not know how to implement it better
            indPatch2Img[min(sel1):max(sel1)+1, min(sel2):max(sel2)+1, iRow] = np.reshape(indPatch[iRow, :, np.newaxis], sB)
            iRow                    += 1

    pInd = collections.namedtuple
    pInd.patch2img = indPatch2Img
    pInd.img2patch = indImg2Patch

    return pInd

def weights_unpatchify(sImg,pInd):
    weights = 1./unpatchify(patchify0(np.ones(sImg), pInd), pInd)
    return weights

# @profile
def patchify0(img,pInd):
    imgFlat = img.flat
   # imgFlat = img.flatten()
    ind = pInd.img2patch.tolist()
    patches = imgFlat[ind]

    return patches

# @profile
def unpatchify(patches,pInd):
    # add a row of zeros to the patches matrix
    h,w = patches.shape
    patchesWithCol = np.zeros([h+1,w])
    patchesWithCol[:-1,:] = patches
    patchesWithColFlat = patchesWithCol.flat
   #  patchesWithColFlat = patchesWithCol.flatten()
    ind = pInd.patch2img.tolist()
    img = np.sum(patchesWithColFlat[ind],axis=2)
    return img


def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


if __name__ =='__main__':
    img = np.random.randint(255,size=[100,100])
    sImg = img.shape
    sP = np.array([39,39])  # size of patch
    step = np.array([1,1])  # sliding window step size
    pInd = get_indices_for_un_patchify(sImg,sP,step)
    patches = patchify0(img,pInd)
    imgOut = unpatchify(patches,pInd)
    weights = weights_unpatchify(sImg,pInd)
    imgOut = weights*imgOut

    print 'Difference of img and imgOut = %.7f' %sum(img.flatten() - imgOut.flatten())
