from PIL import Image
import scipy.io

def readDepthMap(matfilename):
    matfile = scipy.io.loadmat(matfilename)
    depthMapArray = matfile['depthMap']
    depthMap = Image.fromarray(depthMapArray)
    return depthMap
