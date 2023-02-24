import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def init(N: int, seed: int = 0):
    """Construct initial conditions as in the paper "Modelling opinion dynamics under the impact of 
    influencer and media strategies" with M = 2 media and L = 4 influencers.

    Keyword arguments:
    N -- number of individuals (int)
    seed -- seed for the random number generator (int)

    Returns:
    x0 -- initial 2D opinions of N individuals (np.ndarray N x 2)
    y0 -- initial 2D opinions of M = 2 media (np.ndarray 2 x 2)
    z0 -- initial 2D opinions of L = 4 influencers (np.ndarray 4 x 2)
    A -- adjacency matrix of the network between individuals (np.ndarray N x N)
    B -- adjacency matrix of the network between individuals and media (np.ndarray N x 2)
    C0 -- initial adjacency matrix of the network between individuals and influencers (np.ndarray N x 4)   
    """

    np.random.seed(seed)

    # number of media
    M = 2
    # number of influencers
    L = 4

    # individuals' opinions are uniformly distributed in [-2,2] x [-2,2]
    x0 = np.random.rand(N,2)*4 -2 
    # media opinions given by (-1,-1) and (1,1)
    y0 = np.array([[-1., -1.],[1., 1.]])

    # assign each individual to the influencer that is in the
    # same quadrant of the opinion domain
    follinf1 = [i for i in range(N) if x0[i,0]>0 and x0[i,1]>0]   
    follinf2 = [i for i in range(N) if x0[i,0]<=0 and x0[i,1]>0]
    follinf3 = [i for i in range(N) if x0[i,0]>0 and x0[i,1]<=0]
    follinf4 = [i for i in range(N) if x0[i,0]<=0 and x0[i,1]<=0]
    follinf = [follinf1, follinf2, follinf3, follinf4]

    # network between individuals and influencers
    C0 = np.zeros((N, L))
    for i in range(L):
        C0[follinf[i],i] = 1

    # initial opinion of influencer is given by the average opinion 
    # of the individuals that follow them 
    z0 = np.zeros((L,2))
    for i in range(L):
        if len(follinf[i])>0:
            z0[i,:] = x0[follinf[i]].sum(axis = 0)/len(follinf[i])

    # randomly assign media to individuals
    B = np.zeros((N, M))
    assignedmed = np.random.choice([0,1],N)
    B[np.where(assignedmed == 0), 0] = 1
    B[np.where(assignedmed == 1), 1] = 1

    # initialization of fully-connected interaction network 
    # between individuals without self-interactions
    A = np.ones((N,N))-np.diag(np.ones(N))

    return x0, y0, z0, A, B, C0



class opinions:
    """Class to construct, simulate and plot the opinion model with influencers and media for a given 
    parameter set and given initial conditions.
    
    Reference: Helfmann, Luzie, et al. "Modelling opinion dynamics under the impact of influencer and 
    media strategies." preprint arXiv:2301.13661 (2023).
    """
    def __init__(self,  x0: np.ndarray, y0: np.ndarray, z0: np.ndarray, A: np.ndarray, B: np.ndarray, C0: np.ndarray,
                 a: float=1., b: float=2., c: float=4., sigma: float=0.5, sigmahat: float = 0., sigmatilde: float = 0., gamma: float=10., 
                 Gamma: float=100., eta: float = 15., r = lambda x : np.max([0.1,-1+2*x]), phi = lambda x : np.exp(-x),
                 psi = lambda x : np.exp(-x), dt: float = 0.01, domain: np.ndarray = np.array([[-2,2],[-2,2]])):
        """Construct the model class with the given parameters and initial conditions.

        Keyword arguments:
        x0 -- initial 2D opinions of N individuals (np.ndarray N x 2)
        y0 -- initial 2D opinions of M media (np.ndarray M x 2)
        z0 -- initial 2D opinions of L influencers (np.ndarray L x 2)
        A -- adjacency matrix of the network between individuals (np.ndarray N x N)
        B -- adjacency matrix of the network between individuals and media (np.ndarray N x M)
        C0 -- initial adjacency matrix of the network between individuals and influencers (np.ndarray N x L)
        a -- interaction strength between individuals (float)
        b -- interaction strength of media on individuals (float)
        c -- interaction strenght of influencers on individuals (float)
        sigma -- noise strength of individuals' opinion dynamics (float, >=0)
        sigmahat -- noise strength of media's dynamics (float, >=0)
        sigmatilde -- noise strength of influencers dynamics (float, >=0)
        gamma -- inertia paramter of influencers (float, >0)
        Gamma -- inertia parameter of media (float, >0)
        eta -- rate constant of switching influencers (float, >0)
        r -- recommender system function (function)
        phi -- pairwise interaction function between individuals (function)
        psi -- pairwise function when individuals evaluate the suitability of influencers (function)
        dt -- time step size (float, >=0)
        domain -- 2D opinion domain (np.ndarray 2 x 2) assumed to be square
        """

        # initial conditions
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.A = A
        self.B = B
        self.C0 = C0

        # model parameters
        self.N = np.size(x0,0)
        self.M = np.size(y0,0) 
        self.L = np.size(z0,0)
        self.a = a
        self.b = b 
        self.c = c
        self.sigma = sigma 
        self.sigmahat = sigmahat 
        self.sigmatilde = sigmatilde 
        self.gamma = gamma 
        self.Gamma = Gamma 
        self.eta = eta 
        self.r = r 
        self.phi = phi 
        self.psi = psi 
        self.dt = dt 
        self.domain = domain

        # consistency checks
        assert np.shape(self.A) == (self.N, self.N), \
            "The size of the adjacency matrix A does not correspond to the number of individuals N, it should be of the size N x N."
        assert np.shape(self.B) == (self.N, self.M), \
            "The shape of the matrix B should be N x M."
        assert np.shape(self.C0) == (self.N, self.L), \
            "The shape of the matrix C should be N x L."

    def attraction(self, weights: np.ndarray, opinions: np.ndarray, neighbourops: np.ndarray) -> np.ndarray:
        """ Constructs the attraction force on individuals with current opinion given by opinion and between other
         agents with opinions given by neighbourops. The weights array gives the corresponding interaction weights 
         between each individual and each agent. Returns the force."""

        force = np.zeros((self.N,2))
        weightssum = np.sum(weights, axis=1)
        force = weights.dot(neighbourops)/weightssum[:,np.newaxis]
        # the force is zero in case an individual is not interacting with any other agent
        force[weightssum == 0, :] = 0 

        return force-opinions

    def changeinfluencer(self, C: np.ndarray, B: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """ Given the network between individuals and influencers, C, let individuals change their influencer according
        to the specified change rates. Returns the new network. """

        # fraction of individuals following a certain influencer and medium
        fraction = np.zeros((self.M,self.L))
        for i in range(self.L):
            for j in range(self.M):
                fraction[j,i]= B[:,j].dot(C[:,i])/self.N
        # normalized fraction
        normfraction = fraction/np.sum(fraction,axis = 0)[np.newaxis,:]

        # compute distance of indivdiuals to influencers
        dist = cdist(x,z,'euclidean')
        wdist = self.psi(dist) # evaluate the pair function psi on the distances
        
        changerate = np.zeros(self.L)
        for j in range(self.N):
            m = int(B[j,:].dot(range(self.M))) # index of individual j's medium
            # compute change rate that an individuals has to the different influencers
            for l in range(self.L):
                changerate[l] = self.eta * wdist[j,l] *self.r(normfraction[m,l]) 

            # check whether the influencer is changed    
            r1 = np.random.rand() 
            totalrate = np.nansum(changerate)
            if r1 < 1-np.exp(-totalrate*self.dt): # a change event happens in dt-timestep
                prob = changerate/totalrate # probabilities of changing to different influencers
                l = np.random.choice(range(self.L), p=prob)
                # adapt network
                C[j,:] = 0    
                C[j,l] = 1
        return C
            
    def iter(self, x: np.ndarray,y: np.ndarray,z: np.ndarray,C: np.ndarray):
        """ One iteration with step size dt of the opinion model. """

        # media opinions change very slowly based on opinions of followers with friction
        for m in range(self.M):
            Nfoll = np.sum(self.B[:,m])
            if  Nfoll>0:
                averageopinion = self.B[:,m].dot(x)/Nfoll # of followers
                # Euler Mayurama discretization of the SDE
                y[m,:] = y[m,:]  + (self.dt * (averageopinion -y[m,:]) + np.sqrt(self.dt)*self.sigmahat*np.random.randn(2))/self.Gamma

        # influencer opinions adapt slowly to opinions of followers with friction
        for l in range(self.L):
            Nfoll = np.sum(C[:,l])
            if  Nfoll>0:
                averageopinion = C[:,l].dot(x)/Nfoll # of follwers
                z[l,:] = z[l,:]  + (self.dt * (averageopinion -z[l,:]) + np.sqrt(self.dt)*self.sigmatilde*np.random.randn(2))/self.gamma    

        # opinions change due to attracting opinions of friends, influencers and media
        weights = np.multiply(self.A,self.phi(squareform(pdist(x,'euclidean')))) # multiply A and phi entries element-wise
        force = self.a* self.attraction(weights, x, x) + self.b* self.attraction(self.B, x, y) + self.c* self.attraction(C, x, z)
        x = x + self.dt*force + np.sqrt(self.dt)*self.sigma*np.random.randn(self.N,2)

        # boundary condition: don't allow agents to escape domain
        # assumes square domain
        ind1 = np.where(x>self.domain[0,1])  
        ind2 = np.where(x<self.domain[0,0])
        x[ind1] = self.domain[0,1]
        x[ind2] = self.domain[0,0]

        # individuals may change the influencer they are interacting with
        C = self.changeinfluencer( C, self.B, x, z)

        return x,y,z,C
    
    def run(self,timesteps: int = 200, seed: int=0):
        """Simulation of the opinion model for several time steps.

        Keyword arguments:
        timesteps -- number of time steps to simulation (int)
        seed -- seed for the random number generator (int)

        Returns:
        xs -- time-iterated opinions of individuals (list[np.ndarray] of length timesteps + 1)
        ys -- time-iterated opinions of media (list[np.ndarray] of length timesteps + 1)
        zs -- time-iterated opinions of influencers (list[np.ndarray] of length timesteps + 1)
        Cs -- time-iterated network between individuals and influencers (list[np.ndarray] of length timesteps + 1)

        """
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

    def plotsnapshot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, B: np.ndarray, C: np.ndarray, 
                     path: str ="", title: str="", save: bool = False, name: str = "\snapshot.jpg"):
        """Plots a given snapshot of the opinion dynamics as specified by the state (x,y,z,B,C)."""

        fig,ax = plt.subplots()

        colors = ["#44AA99","#DDCC77","#CC6677","#88CCEE", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        if np.size(colors)<self.L:
            "There are not enough colors specified for the number of influencers."
        markers = ["o", "^", "D", "+"]
        if np.size(markers)<self.M:
            "There are not enough markers specified for the number of media."
        
        for l in range(self.L):
            for m in range(self.M):
                indices = np.where(B[:,m]*C[:,l]==1) # of individuals that are attached to influencer l and medium m
                ax.scatter(x[indices,0],x[indices,1], c=colors[l], marker = markers[m], s = 25,alpha=0.8)
        
        for l in range(self.L):
            ax.scatter(z[l,0], z[l,1],c=colors[l], s = 50,edgecolor='k')
        for m in range(self.M):
            ax.scatter(y[m,0], y[m,1],marker = markers[m], c='k', s = 50)

        plt.xlim(self.domain[0,:])
        plt.ylim(self.domain[1,:])
        plt.title(title)
        if save==True:
            fig.savefig(path+name, format='jpg', dpi=200, bbox_inches='tight')

        plt.close()
    
    def makegif(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, Cs: np.ndarray, gifpath: str="", framespath: str="", 
                name: str ="\\realization.gif", stepsize: int = 5, fps: int = 5):
        """ Makes a gif of the realization specified by (xs,ys,zs,Cs,B), the frames for the gif are safed in framespath while the 
        final gif is stored under gifpath+name."""

        gifpath = gifpath+name
        framespath = framespath
        name = "\{i}.jpg"

        times = range(0,np.size(xs,0),stepsize)

        for index, t in enumerate(times):
            self.plotsnapshot(xs[t], ys[t], zs[t], self.B, Cs[t],title="t = "+str(np.round(self.dt*t,2)),save=True, path = framespath, name = name.format(i=index))
 
        with imageio.get_writer(gifpath , mode='I',fps = fps) as writer:
            for index, t in enumerate(times):
                writer.append_data(imageio.imread(framespath+name.format(i=index)))


