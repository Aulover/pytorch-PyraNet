import numpy as np
import cv2
import ref as ref
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
try:
    #import mayavi.mlab
    pass
except:
    pass
    
def show2D(img, points, c):
    points = ((points.reshape(ref.nJoints, -1))).astype(np.int32)
    for j in range(ref.nJoints):
        cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
    for e in ref.edges:
        cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                                    (points[e[1], 0], points[e[1], 1]), c, 2)
    return img

class Debugger(object):
    def __init__(self):
        self.plt = plt
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot((111),projection='3d')
        self.ax.grid(False)
        #self.ax.set_xlabel('x') 
        #self.ax.set_ylabel('y') 
        #self.ax.set_zlabel('z')
        oo = 1e10
        self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
        self.xmin, self.ymin, self.zmin = oo, oo, oo
        self.imgs = {}
        self.vols = {}
        try:
            self.mayavi = mayavi
            xx = [0, 0, 0, 0, ref.outputRes, ref.outputRes, ref.outputRes, ref.outputRes]
            yy = [0, 0, ref.outputRes, ref.outputRes, 0, 0, ref.outputRes, ref.outputRes]
            zz = [0, ref.outputRes, 0, ref.outputRes, 0, ref.outputRes, 0, ref.outputRes]
            self.mayavi.mlab.points3d(xx, yy, zz,
                                                     mode = "cube",
                                                     color = (0, 0, 0),
                                                     opacity = 1,
                                                     scale_factor=1)
        except:
            pass
        
    def addImg(self, img, imgId = 'default'):
        self.imgs[imgId] = img.copy()
    
    def addVol(self, vol, c = (0, 1, 0), threshold = 0.5, volID = 'default'):
        self.vols[volID] = vol.copy()
        zz, yy, xx = np.where(self.vols[volID] > threshold)
        self.mayavi.mlab.points3d(xx, ref.outputRes - yy, zz,
                                                 mode = "cube",
                                                 color = c,
                                                 opacity = 0.1,
                                                 scale_factor=1)
    
    def addPoint2D(self, point, c, imgId = 'default'):
        self.imgs[imgId] = show2D(self.imgs[imgId], point, c)
    
    def showImg(self, pause = False, imgId = 'default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()
    
    def showVol(self):
        self.mayavi.mlab.show()
    
    def showAllImg(self, pause = False):
        for i, v in self.imgs.items():
            cv2.imshow('{}'.format(i), v)
        if pause:
            cv2.waitKey()
    
    def saveImg(self, imgId = 'default', path = '../debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
        
    def saveAllImg(self, path = '../debug/'):
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}.png'.format(i), v)
        
