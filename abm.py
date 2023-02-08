import numpy as np
from scipy.spatial.distance import cdist, pdist,squareform

# Todo make class: initial conditions are outside and given when defining model with parameters in init
# set random seeds

def parameters():
    N=250 # number of individuals
    M=2 # number of media
    L=4 # number of influencers
    a=1. # interaction strength between individuals
    b=2. # interactions strength between media and individuals
    c=4. # interaction strength between individuals and influencers
    sigma = 0.5 # noise on individual agents
    sigmahat = 0.# noise on media
    sigmatilde = 0. # noise on influencers
    gamma = 10. # friction for influencers
    Gamma = 100. # friction for media
    eta = 15. # rate constant of switching influencer
    r = lambda x : np.max([0.1,-1+2*x]) # recommender system function
    phi = lambda x : np.exp(-x) # pair function  between individuals
    psi = lambda x : np.exp(-x) # pair functions for switching influencer
    dt = 0.01
    domain = np.array([[-2,2],[-2,2]]) #quadratic
    p = {'N':N, 'M': M, 'L':L, 'a':a, 'b':b,'c':c, 'sigma':sigma, 'sigmahat':sigmahat, 'sigmatilde':sigmatilde, 'r':r, 'eta':eta, 'phi':phi, 'psi':psi, 'gamma':gamma, 'Gamma':Gamma, 'dt':dt, 'domain':domain} 
    return p

def init(p):
    N = p['N']
    M = p['M']
    L = p['L']
    if M !=2:
        "These initial conditions are for the case of 2 media."
    if L != 4: 
        "These initial conditions are for the case of 4 influencers."
    # individuals' opinions
    x = np.random.rand(N,2)*4 -2 
    # media opinions
    y = np.array([[-1., -1.],[1., 1.]])
    # assign individuals to different influencer depending 
    # on the different quadrants they start in
    follinf1 = [i for i in range(N) if x[i,0]>0 and x[i,1]>0]   
    follinf2 = [i for i in range(N) if x[i,0]<=0 and x[i,1]>0]
    follinf3 = [i for i in range(N) if x[i,0]>0 and x[i,1]<=0]
    follinf4 = [i for i in range(N) if x[i,0]<=0 and x[i,1]<=0]
    follinf = [follinf1, follinf2, follinf3, follinf4]
    # network between individuals and influencers
    C=np.zeros((N, L))
    for i in range(L):
        C[follinf[i],i] =1
    # initial opinions of influencers given by average follower opinion
    z = np.zeros((L,2))
    for i in range(L):
        if len(follinf[i])>0:
            z[i,:] = x[follinf[i]].sum(axis = 0)/len(follinf[i])
    # randomly assign medium
    B=np.zeros((N, M))
    assignedmed = np.random.choice([0,1],N)
    B[np.where(assignedmed==0), 0] = 1
    B[np.where(assignedmed==1), 1] = 1
    # initialization of interaction network between individuals
    # without self-interactions
    A = np.ones((N,N))-np.diag(np.ones(N))
    return x,y,z,A,B,C

def attraction(weights, positions, neighbourpos,p):
    N = p['N']
    force = np.zeros((N,2))
    for i in range(N):
        weightssum = np.sum(weights[i,:])
        if weightssum==0:
            force[i,:]=np.array([0, 0])
        else:
            force[i,:] = (1/weightssum)* weights[i,:].dot(neighbourpos)
    return force-positions

def changeinfluencer(p, C, B,x, z):
    eta = p['eta']
    N = p['N']
    M = p['M']
    L = p['L']
    dt = p['dt']
    r = p['r']
    psi = p['psi']
    # fraction of individuals following a certain influencer and medium
    fraction = np.zeros((M,L))
    for i in range(L):
        for j in range(M):
            fraction[j,i]= B[:,j].dot(C[:,i])/N
    normfraction = fraction/np.sum(fraction,axis = 0)[np.newaxis,:]
    # compute distance of followers to influencers
    dist = cdist(x,z,'euclidean')
    wdist = psi(dist)
    # compute attractiveness of influencer to individuals
    attractiveness = np.zeros(L)
    for j in range(N):
        for l in range(L):
            m = int(B[j,:].dot(range(M))) # index of individual j's medium
            attractiveness[l]= eta * wdist[j,l] *r(normfraction[m,l]) 
        r1=np.random.rand()
        totalrate = np.nansum(attractiveness)
        if r1<1-np.exp(-totalrate*dt): # influencer is changed
            prob = attractiveness/totalrate #probability with which influencer is changed
            r2=np.random.rand()
            l=1
            while np.nansum(prob[0:l])<r2: # choose to which influencer to change to
                l=l+1
            # adapt network
            C[j,:]=np.zeros((L))      
            C[j,l-1]=1
    return C
        
def iter(x,y,z,A,B,C,p):
    phi = p['phi']   
    dt = p['dt']    
    Gamma = p['Gamma']    
    gamma = p['gamma']
    sigma = p['sigma'] 
    sigmatilde = p['sigmatilde'] 
    sigmahat = p['sigmahat'] 
    domain = p['domain'] 
    N = p['N']
    a = p['a']
    b = p['b']
    c = p['c']
    M = p['M']
    L = p['L']
    # media opinions change very slowly based on opinions of followers with friction
    for m in range(M):
        Nfoll = np.sum(B[:,m])
        if  Nfoll>0:
            averageopinion = B[:,m].dot(x)/Nfoll
            y[m,:] = y[m,:]  + (dt * (averageopinion -y[m,:]) + np.sqrt(dt*sigmahat)*np.random.randn(2))/Gamma
    # influencer opinions adapt slowly to opinions of followers with friction
    for l in range(L):
        Nfoll = np.sum(C[:,l])
        if  Nfoll>0:
            averageopinion = C[:,l].dot(x)/Nfoll
            z[l,:] = z[l,:]  + (dt * (averageopinion -z[l,:]) + np.sqrt(dt*sigmatilde)*np.random.randn(2))/gamma        
    # opinions change due to attracting opinions of friends, influencers and media
    weights = np.multiply(A,phi(squareform(pdist(x,'euclidean')))) #multiply A and phi entries element-wise
    forceinds= a* attraction(weights, x, x, p)
    forcemedia= b* attraction(B, x, y,p)
    forceinfs = c* attraction(C, x, z,p)
    force = forceinds + forcemedia + forceinfs
    x = x + dt*force + np.sqrt(dt*sigma)*np.random.randn(N,2)
    # boundary condition: don't allow agents to escape domain
    ind1 = np.where(x>domain[0,1])  
    ind2 = np.where(x<domain[0,0])
    x[ind1] = domain[0,1]
    x[ind2] = domain[0,0]
    # individuals may change the influencer they are interacting with
    C = changeinfluencer(p, C, B, x, z)
    return x,y,z,C

 