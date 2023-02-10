import numpy as np
from scipy.spatial.distance import cdist, pdist,squareform
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# add test
# time and make fast

def init(N,seed = 0):
    np.random.seed(seed)
    M = 2
    L = 4

    # individuals' opinions
    x0 = np.random.rand(N,2)*4 -2 
    # media opinions
    y0 = np.array([[-1., -1.],[1., 1.]])

    # assign individuals to different influencer depending 
    # on the different quadrants they start in
    follinf1 = [i for i in range(N) if x0[i,0]>0 and x0[i,1]>0]   
    follinf2 = [i for i in range(N) if x0[i,0]<=0 and x0[i,1]>0]
    follinf3 = [i for i in range(N) if x0[i,0]>0 and x0[i,1]<=0]
    follinf4 = [i for i in range(N) if x0[i,0]<=0 and x0[i,1]<=0]
    follinf = [follinf1, follinf2, follinf3, follinf4]

    # network between individuals and influencers
    C0=np.zeros((N, L))
    for i in range(L):
        C0[follinf[i],i] =1

    # initial opinions of influencers given by average follower opinion
    z0 = np.zeros((L,2))
    for i in range(L):
        if len(follinf[i])>0:
            z0[i,:] = x0[follinf[i]].sum(axis = 0)/len(follinf[i])

    # randomly assign medium
    B=np.zeros((N, M))
    assignedmed = np.random.choice([0,1],N)
    B[np.where(assignedmed==0), 0] = 1
    B[np.where(assignedmed==1), 1] = 1

    # initialization of interaction network between individuals
    # without self-interactions
    A = np.ones((N,N))-np.diag(np.ones(N))

    return x0,y0,z0,A,B,C0



class opinions:
    """Model class."""
    def __init__(self,  x0, y0, z0, A, B, C0, N=250, M=2, L=4, a=1, b=2, c=4, sigma=0.5, sigmahat = 0, sigmatilde = 0, 
                 gamma=10, Gamma=100, eta = 15, r = lambda x : np.max([0.1,-1+2*x]), phi = lambda x : np.exp(-x),
                 psi = lambda x : np.exp(-x), dt = 0.01, domain = np.array([[-2,2],[-2,2]])):
        
        # initial conditions
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.A = A
        self.B = B
        self.C0 = C0

        # model parameters
        self.N = N # number of individuals
        self.M = M # number of media
        self.L = L # number of influencers
        self.a = a # interaction strength between individuals
        self.b = b # interactions strength between media and individuals
        self.c = c # interaction strength between individuals and influencers
        self.sigma = sigma # noise on individual agents
        self.sigmahat = sigmahat # noise on media
        self.sigmatilde = sigmatilde # noise on influencers
        self.gamma = gamma # friction for influencers
        self.Gamma = Gamma # friction for media
        self.eta = eta # rate constant of switching influencer
        self.r = r # recommender system function
        self.phi = phi # pair function  between individuals
        self.psi = psi # pair functions for switching influencer
        self.dt = dt
        self.domain = domain

        # consistency check
        assert self.N == np.size(x0,0), \
            "The size of the initial state of individuals does not align with the parameter N."
        assert self.M == np.size(y0,0), \
            "The size of the initial state of media does not align with the parameter M."
        assert self.L == np.size(z0,0), \
            "The size of the initial state of influencers does not align with the parameter L."

    def attraction(self,weights, positions, neighbourpos):
        force = np.zeros((self.N,2))
        for i in range(self.N):
            weightssum = np.sum(weights[i,:])
            if weightssum==0:
                force[i,:]=np.array([0, 0])
            else:
                force[i,:] = (1/weightssum)* weights[i,:].dot(neighbourpos)
        return force-positions

    def changeinfluencer(self, C, B,x, z):

        # fraction of individuals following a certain influencer and medium
        fraction = np.zeros((self.M,self.L))
        for i in range(self.L):
            for j in range(self.M):
                fraction[j,i]= B[:,j].dot(C[:,i])/self.N
        normfraction = fraction/np.sum(fraction,axis = 0)[np.newaxis,:]

        # compute distance of followers to influencers
        dist = cdist(x,z,'euclidean')
        wdist = self.psi(dist)

        
        suitability = np.zeros(self.L)
        for j in range(self.N):
            # compute suitability of each influencer to individual j
            for l in range(self.L):
                m = int(B[j,:].dot(range(self.M))) # index of individual j's medium
                suitability[l]= self.eta * wdist[j,l] *self.r(normfraction[m,l]) 

            #check whether the influencer is changed    
            r1=np.random.rand()
            totalrate = np.nansum(suitability)
            if r1<1-np.exp(-totalrate*self.dt): 
                prob = suitability/totalrate  
                r2 = np.random.rand()
                l = 1
                while np.nansum(prob[0:l])<r2: # choose to which influencer to change to
                    l = l+1

                # adapt network
                C[j,:] = 0    
                C[j,l-1] = 1
        return C
            
    def iter(self,x,y,z,C):

        # media opinions change very slowly based on opinions of followers with friction
        for m in range(self.M):
            Nfoll = np.sum(self.B[:,m])
            if  Nfoll>0:
                averageopinion = self.B[:,m].dot(x)/Nfoll
                y[m,:] = y[m,:]  + (self.dt * (averageopinion -y[m,:]) + np.sqrt(self.dt*self.sigmahat)*np.random.randn(2))/self.Gamma

        # influencer opinions adapt slowly to opinions of followers with friction
        for l in range(self.L):
            Nfoll = np.sum(C[:,l])
            if  Nfoll>0:
                averageopinion = C[:,l].dot(x)/Nfoll
                z[l,:] = z[l,:]  + (self.dt * (averageopinion -z[l,:]) + np.sqrt(self.dt*self.sigmatilde)*np.random.randn(2))/self.gamma    

        # opinions change due to attracting opinions of friends, influencers and media
        weights = np.multiply(self.A,self.phi(squareform(pdist(x,'euclidean')))) # multiply A and phi entries element-wise
        forceinds= self.a* self.attraction(weights, x, x)
        forcemedia= self.b* self.attraction(self.B, x, y)
        forceinfs = self.c* self.attraction(C, x, z)
        force = forceinds + forcemedia + forceinfs
        x = x + self.dt*force + np.sqrt(self.dt*self.sigma)*np.random.randn(self.N,2)

        # boundary condition: don't allow agents to escape domain
        # assumes square domain
        ind1 = np.where(x>self.domain[0,1])  
        ind2 = np.where(x<self.domain[0,0])
        x[ind1] = self.domain[0,1]
        x[ind2] = self.domain[0,0]

        # individuals may change the influencer they are interacting with
        C = self.changeinfluencer( C, self.B, x, z)

        return x,y,z,C
    
    def run(self,timesteps = 200, seed=0):
        np.random.seed(seed)
        x=self.x0.copy()
        xs = [x.copy()]
        y=self.y0.copy()
        ys = [y.copy()]
        z=self.z0.copy()
        zs = [z.copy()]
        C = self.C0.copy()
        Cs = [C.copy()]

        for t in range(timesteps):
            x,y,z,C = self.iter(x,y,z,C)
            xs.append(x.copy())
            ys.append(y.copy())
            zs.append(z.copy())
            Cs.append(C.copy())

        return xs,ys,zs,Cs  

    def plotsnapshot(self, x, y, z, B, C, path=None, title="",save = False, name = "\snapshot.jpg"):
        fig,ax = plt.subplots()

        colors = ["#44AA99","#DDCC77","#CC6677","#88CCEE", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        if np.size(colors)<self.L:
            "There are not enough colors specified for the number of influencers."
        markers = ["o", "^", "D", "+"]
        if np.size(markers)<self.M:
            "There are not enough markers specified for the number of media."
        
        for l in range(self.L):
            ax.scatter(z[l,0], z[l,1],c=colors[l], s = 40,edgecolor='k')
        for m in range(self.M):
            ax.scatter(y[m,0], y[m,1],marker = markers[m], c='k', s = 40)

        for l in range(self.L):
            for m in range(self.M):
                indices = np.where(B[:,m]*C[:,l]==1) # of individuals that are attached to influencer l and medium m
                ax.scatter(x[indices,0],x[indices,1], c=colors[l], marker = markers[m], s = 20,alpha=0.8)
        plt.xlim(self.domain[0,:])
        plt.ylim(self.domain[1,:])
        plt.title(title)
        if save==True:
            fig.savefig(path+name, format='jpg', dpi=200, bbox_inches='tight')

        plt.close()
    
    def makegif(self,xs,ys,zs,Cs, gifpath=None, framespath=None, name="\\realization.gif", stepsize = 5,fps = 5):
        gifpath = gifpath+name
        framespath = framespath
        name = "\{i}.jpg"

        times = range(0,np.size(xs,0),stepsize)

        for i in times:
            self.plotsnapshot(xs[i], ys[i], zs[i], self.B, Cs[i],title="t = "+str(np.round(self.dt*i,2)),save=True, path = framespath, name = name.format(i=i))
 
        with imageio.get_writer(gifpath , mode='I',fps = fps) as writer:
            for i in times:
                writer.append_data(imageio.imread(framespath+name.format(i=i)))


