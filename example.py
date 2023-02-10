from SocialMediaModelPy import abm
#import os.path
import time


#mypath = os.path.abspath(os.path.dirname(__file__))
imgpath = "img" #os.path.join(mypath, 'img')
framespath = "img\\frames" #os.path.join(imgpath, 'frames')


N = 250
t1 = time.time()
x0,y0,z0,A,B,C0 = abm.init(N,seed=1)
t2 = time.time()
ops = abm.opinions(x0,y0,z0,A,B,C0)
t3 = time.time()
xs,ys,zs,Cs = ops.run(timesteps=300,seed=1)
t4 = time.time()
ops.plotsnapshot(ops.x0,ops.y0,ops.z0,ops.B,ops.C0,save=True,path=imgpath)
t5 = time.time()
ops.makegif(xs,ys,zs,Cs,stepsize=5,gifpath=imgpath, framespath=framespath)
t6 = time.time()

print("Elapsed time for init = %s" % (t2-t1))
print("Elapsed time for class construction = %s" % (t3-t2))
print("Elapsed time for simulation = %s" % (t4-t3))
print("Elapsed time for a single plot = %s" % (t5-t4))
print("Elapsed time for creating the gif = %s" % (t6-t5))
 
print(type(xs))

