from SocialMediaModelPy import abm
#import os.path

#mypath = os.path.abspath(os.path.dirname(__file__))
imgpath = "img" #os.path.join(mypath, 'img')
framespath = "img\\frames" #os.path.join(imgpath, 'frames')


N = 250
x0,y0,z0,A,B,C0 = abm.init(N,seed=1)
ops = abm.opinions(x0,y0,z0,A,B,C0,N=N)
xs,ys,zs,Cs = ops.run(timesteps=300,seed=1)
ops.plotsnapshot(ops.x0,ops.y0,ops.z0,ops.B,ops.C0,save=True,path=imgpath)
ops.makegif(xs,ys,zs,Cs,stepsize=5,gifpath=imgpath, framespath=framespath)