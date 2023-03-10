import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def initialcondition(N: int, L: int = 4, seed: int = 0):
    """Construct initial conditions as in the paper "Modelling opinion dynamics under the impact of 
    influencer and media strategies" with M = 2 media and L = 4 influencers.

    Keyword arguments:
    N -- number of individuals (int)
    L -- number of influencers (int)
    seed -- seed for the random number generator (int)

    Returns:
    x0 -- initial 2D opinions of N individuals (np.ndarray N x 2)
    y0 -- initial 2D opinions of M media (np.ndarray M x 2)
    z0 -- initial 2D opinions of L influencers (np.ndarray L x 2)
    A -- adjacency matrix of the network between individuals (np.ndarray N x N)
    B -- adjacency matrix of the network between individuals and media (np.ndarray N x M)
    C0 -- initial adjacency matrix of the network between individuals and influencers (np.ndarray N x L)   
    D -- adjacency matrix of the network between influencers (np.ndarray L x L)
    """

    if L!=4:
        print("The initial condition only works for L=4")

    np.random.seed(seed)

    # individuals' opinions are uniformly distributed in [-2,2] x [-2,2]
    x0 = np.random.rand(N,2)*4 -2 

    follinf = np.zeros(N)
    for i in range(N):
        follinf[i] = np.random.choice(range(0,L), p= np.repeat((1/L),L))
    
    follinf = follinf.astype('int')
    followergroups = np.empty((L,0)).tolist()
    for i in range(N):
        for j in range(L):
            if follinf[i] == j:
                followergroups[j].append(i)
    
    # network between individuals and influencers
    # one 1 in each row 
    C0 = np.zeros((N, L)) 
    #for i in range(L):
    for i in range(N):
        for j in range(L):
            if follinf[i] == j:
                C0[i,j] = 1
            
       # C0[follinf[i] == j] = 1

    # initial opinion of influencer is given by the average opinion 
    # of the individuals that follow them 
    z0 = np.array([[-1.237304  , -0.88463197],
       [ 0.59117453,  1.29053742],
       [ 0.64113249, -0.30845413],
       [-0.65820039, -1.00763623]])#np.random.rand(L,2)*4 -2

    # initialization of fully-connected interaction network 
    # between individuals without self-interactions
    A = np.ones((N,N))-np.diag(np.ones(N))

    #initialize influencer network
    D = np.ones((L,L))-np.diag(np.ones(L)) ##
    
    return x0, z0, A, C0, D ##



class opinions:
    """Class to construct, simulate and plot the opinion model with influencers and media for a given 
    parameter set and given initial conditions.
    
    Reference: Helfmann, Luzie, et al. "Modelling opinion dynamics under the impact of influencer and 
    media strategies." preprint arXiv:2301.13661 (2023).
    """
    def __init__(self,  x0: np.ndarray, z0: np.ndarray, A: np.ndarray, C0: np.ndarray, D: np.ndarray,
                 a: float=0.5, d: float = 0.5, sigma: float=0.1, sigmatilde: float = 0.1, gamma: float=10., #c: float=4., 
                 eta: float = 15.,  psi = lambda x : np.exp(-x), dt: float = 0.01, 
                 domain: np.ndarray = np.array([[-2,2],[-2,2]]), 
                 theta_ind: float = 1.5, theta_inf: float = 1.5,
                 level_off: bool = False, omikron_ind: float = 3.0, omikron_inf: float = 3.0): ##
        """Construct the model class with the given parameters and initial conditions.

        Keyword arguments:
        x0 -- initial 2D opinions of N individuals (np.ndarray N x 2)
        y0 -- initial 2D opinions of M media (np.ndarray M x 2)
        z0 -- initial 2D opinions of L influencers (np.ndarray L x 2)
        A -- adjacency matrix of the network between individuals (np.ndarray N x N)
        B -- adjacency matrix of the network between individuals and media (np.ndarray N x M)
        C0 -- initial adjacency matrix of the network between individuals and influencers (np.ndarray N x L)
        D -- adjacency matrix of the network between influencers (np.ndarray L x L)
        a -- interaction strength between individuals (float)
        b -- interaction strength of media on individuals (float)
        c -- interaction strenght of influencers on individuals (float)
        sigma -- noise strength of individuals' opinion dynamics (float, >=0)
        sigmahat -- noise strength of media's dynamics (float, >=0)
        sigmatilde -- noise strength of influencers dynamics (float, >=0)
        gamma -- inertia paramter of influencers (float, >=1)
        Gamma -- inertia parameter of media (float, >=1)
        eta -- rate constant of switching influencers (float, >0)
        r -- recommender system function (function)
        theta_ind -- interaction threshold between individuals (float, >0)
        theta_inf -- interaction threshold between influencers (float, >0)
        phi-- pairwise interaction function between individuals or between influencers (function)
        psi -- pairwise function when individuals evaluate the suitability of influencers (function)
        dt -- time step size (float, >=0)
        domain -- 2D opinion domain (np.ndarray 2 x 2) assumed to be square
        """
        
        # initial conditions
        self.x0 = x0
        self.z0 = z0
        self.A = A
        self.C0 = C0
        self.D = D

        # model parameters
        self.N = np.size(x0,0)
        self.L = np.size(z0,0)
        self.a = a
        self.c = 1 - self.a 
        self.d = d
        self.e = 1 - self.d

        self.sigma = sigma 
        self.sigmatilde = sigmatilde 
        self.gamma = gamma 
        self.eta = eta 
        self.psi = psi 
        self.dt = dt 
        self.domain = domain
        
        self.theta_ind = theta_ind
        self.theta_inf = theta_inf ##
        self.zeta_ind = 2 * np.log(9)/ self.theta_ind
        self.zeta_inf = 2 * np.log(9)/ self.theta_inf##
        self.omikron_ind = omikron_ind
        self.omikron_inf = omikron_inf
        self.level_off = level_off
        
        # consistency checks
        assert np.shape(self.A) == (self.N, self.N), \
            "The size of the adjacency matrix A does not correspond to the number of individuals N, it should be of the size N x N."
        assert np.shape(self.C0) == (self.N, self.L), \
            "The shape of the matrix C should be N x L."
            
        if a > 1:
            print("a is larger than 1")
            a = 1
            
    def phi(self, x, theta: float, zeta: float):
        return 1 / ( 1 + np.exp( zeta * ( x - theta ) ) ) - 0.5 

    def phi_level_off(self, x, theta: float, zeta: float, omikron: float):
        return (1 / (1 + np.exp(zeta * (x - theta))) - 0.5) + \
            0.5 * (1 / (1 + np.exp(- zeta * (x - (theta + omikron)))))

    def attraction(self, weights: np.ndarray, opinions: np.ndarray, neighbourops: np.ndarray) -> np.ndarray:
        """ Constructs the attraction force on individuals with current opinion given by opinion and between other
         agents with opinions given by neighbourops. The weights array gives the corresponding interaction weights 
         between each individual and each agent. Returns the force."""

        weightssum = np.sum(abs(weights), axis=1)
        weight1 =  weights/weightssum[:,np.newaxis]
        weight2 = np.divide(np.sum(weights, axis=1),weightssum)
        force = weight1.dot(neighbourops) - np.multiply(weight2[:,np.newaxis],opinions)

        # the force is zero in case an individual is not interacting with any other agent
        force[weightssum == 0, :] = 0 

        return force

    def changeinfluencer(self, C: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """ Given the network between individuals and influencers, C, let individuals change their influencer according
        to the specified change rates. Returns the new network. """

        # compute distance of indivdiuals to influencers
        dist = cdist(x,z,'euclidean')
        wdist = self.psi(dist) # evaluate the pair function psi on the distances
        
        changerate = np.zeros(self.L)
        for j in range(self.N):
            # compute change rate that an individuals has to the different influencers
            for l in range(self.L):
                changerate[l] = self.eta * wdist[j,l] 

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
            
    def iter(self, x: np.ndarray, z: np.ndarray, C: np.ndarray):
        """ One iteration with step size dt of the opinion model. """

        # opinions change due to attracting opinions of friends, influencers and media
        if self.level_off == False:
            #weights = np.multiply(self.A, self.phi_ind(squareform(pdist(x,'euclidean')))) # multiply A and phi entries element-wise
            weights = np.multiply(self.A, self.phi(squareform(pdist(x,'euclidean')), self.theta_ind, self.zeta_ind))
            force = self.a* self.attraction(weights, x, x) + self.c* self.attraction(C, x, z)

            #opinion changes of influencers in the direction of average follower and with attraction-repulsion to other influencers
            #weights_inf = np.multiply(self.D,self.phi_inf(squareform(pdist(z,'euclidean')))) # multiply D and phi entries element-wise; define earlier?

            weights_inf = np.multiply(self.D, self.phi(squareform(pdist(z,'euclidean')), self.theta_inf, self.zeta_inf))
            force_inf = self.e*self.attraction(C.T,z,x) + self.d*self.attraction(weights_inf, z, z) #define earlier?

        else:

            weights = np.multiply(self.A, self.phi_level_off(squareform(pdist(x, 'euclidean')), self.theta_ind, self.zeta_ind, self.omikron_ind))
            force = self.a * self.attraction(weights, x, x) + self.c * self.attraction(C, x, z)

            weights_inf = np.multiply(self.D, self.phi_level_off(squareform(pdist(z,'euclidean')), self.theta_inf, self.zeta_inf, self.omikron_inf))
            force_inf = self.e*self.attraction(C.T,z,x) + self.d*self.attraction(weights_inf, z, z) #define earlier?


        z = z + self.dt*force_inf + np.sqrt(self.dt)*self.sigmatilde*np.random.randn(self.L,2)/self.gamma
       
        x = x + self.dt*force + np.sqrt(self.dt)*self.sigma*np.random.randn(self.N,2)

        # boundary condition: don't allow agents to escape domain
        # assumes square domain
        ind1 = np.where(x>self.domain[0,1])  
        ind2 = np.where(x<self.domain[0,0])
        x[ind1] = self.domain[0,1]
        x[ind2] = self.domain[0,0]

        ind1 = np.where(z>self.domain[0,1])  
        ind2 = np.where(z<self.domain[0,0])
        z[ind1] = self.domain[0,1]
        z[ind2] = self.domain[0,0]

        # individuals may change the influencer they are interacting with
        C = self.changeinfluencer(C,  x, z)

        return x,z,C
    
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
        z=self.z0.copy()
        zs = [z.copy()]
        C = self.C0.copy()
        Cs = [C.copy()]

        for t in range(timesteps):
            x,z,C = self.iter(x,z,C)
            xs.append(x.copy())
            zs.append(z.copy())
            Cs.append(C.copy())

        return xs,zs,Cs  

    def plotsnapshot(self, x: np.ndarray, z: np.ndarray, C: np.ndarray, 
                     path: str ="", title: str="", save: bool = False,name: str = "/snapshot.jpg"): ##
        """Plots a given snapshot of the opinion dynamics as specified by the state (x,y,z,B,C)."""

        fig,ax = plt.subplots()

        colors = ["#44AA99","#DDCC77","#CC6677","#88CCEE", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        if np.size(colors)<self.L:
            "There are not enough colors specified for the number of influencers."
        
        for l in range(self.L):
            indices = np.where(C[:,l]==1) # of individuals that are attached to influencer l and medium m
            ax.scatter(x[indices,0],x[indices,1], c=colors[l],  s = 25,alpha=0.8)
        
        for l in range(self.L):
            ax.scatter(z[l,0], z[l,1],c=colors[l], s = 50, edgecolor='k')


        plt.xlim(self.domain[0,:])
        plt.ylim(self.domain[1,:])
        plt.title(title)
        if save==True:
            #name = "/snapshot.jpg"##
            fig.savefig(path+name, format='jpg', dpi=200, bbox_inches='tight')

        plt.close()


    def makegif(self, xs: np.ndarray,  zs: np.ndarray, Cs: np.ndarray, gifpath: str="", framespath: str="",
                gifname: str="", stepsize: int = 5, fps: int = 5): ##
        """ Makes a gif of the realization specified by (xs,ys,zs,Cs,B), the frames for the gif are safed in framespath while the
        final gif is stored under gifpath+name."""

        if gifname == "":
            name = "/realization.gif"#
        else:
            name = gifname
        gifpath = gifpath+name
        framespath = framespath
        name = "/{i}.jpg" ##

        times = range(0,np.size(xs,0),stepsize)

        for index, t in enumerate(times):
            self.plotsnapshot(xs[t],  zs[t], Cs[t],title="t = "+str(np.round(self.dt*t,2)),save=True, path = framespath, name = name.format(i=index))

        with imageio.get_writer(gifpath , mode='I',fps = fps) as writer:
            for index, t in enumerate(times):
                writer.append_data(imageio.imread(framespath+name.format(i=index)))

