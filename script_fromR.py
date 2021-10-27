def createDATA1(DAT,j,W,N,col,dd,angle,add,name,logSCALE,dens, dominion, logPlot):

    # input:
    # DAT risposte a seed e target questions
    # W1 vettore dei pesi di Cooke per target
    # W2 vettore dei pesi di Cooke per seed
    # W3 vettore pesi uniformi
    # j indice della domanda
    # N non si cambia mai
    # logscale si potrebbe anche prendere dal file DAT
    # dominion dice che campioni eliminare
    
    
    
    import numpy as np

    n = int(np.amax(DAT[:,0]))
    nn = int(len(DAT[:,0])/n)
    
    incm = DAT[np.arange(n)*nn+j,2]
    mid = DAT[np.arange(n)*nn+j,3]
    incM = DAT[np.arange(n)*nn+j,4]
    

    quan05,quan50,qmean,quan95,C = createSamplesUCA2(incm,mid,incM,W,N,col,dd,angle,add,name,logSCALE,dens,dominion,logPlot)

    return quan05,quan50,qmean,quan95,C

def max_entropy(incm,mid,incM,rA,rB):

    import numpy as np

    rng = np.random.default_rng()
    x = rng.random(1)
    y = mid+(incM-mid)*rng.random(1)
    
    if(x>0.95):
    
        y = incM+(rB-incM)*rng.random(1)

    if(x<0.5):
        
        y = incm+(mid-incm)*rng.random(1)
        
    if(x<0.05):
   
        y = rA+(incm-rA)*rng.random(1)
    
    return y 
    
def sampleDISCR(P,N):

    import numpy as np

    n = len(P)
    PP = np.zeros(n)
    
    PP[0] = P[0]
    
    for i in np.arange(1,n-1):
    
        PP[i] = PP[i-1]+P[i]
    
    PP[n-1]=1
    #print(PP)
    
    rng = np.random.default_rng(12345)
    u = rng.random(N)
    a = np.zeros(N)
    
    for i in np.arange(N):
    
        B = u[i]-PP
        B = B[B<=0]
        b = len(B)
        a[i] = n-b+1
    
        #print(c(u[i],a[i]))

    return a
    
    
def createSamplesUCA2(incm,mid,incM,W,N,col,dd,angle,add,name,logSCALE,dens,dominion,logPlot):

    import numpy as np

    #print(incm)
    #print(mid)
    #print(incM)
    
    
    W = W/np.sum(W)
    
    if (logSCALE):
    
        VV = incm>0
        incm = np.log(incm[VV]) / np.log(10)
        incM = np.log(incM[VV]) / np.log(10)
        mid = np.log(mid[VV]) / np.log(10)
        W = W[VV]
        W = W/sum(W)
        
    C = np.zeros(N)
    Ne = len(incm)
    rA = np.amin(incm[W>0])
    rB = np.amax(incM[W>0])
    R = rB-rA
    rA = rA-R/10
    rB = rB+R/10
                
    # write(c(rA,rB),paste0(name,'RANGE.asc'),ncolumns=1)
    
    sV = sampleDISCR(W,N)
    
    # write(sV,paste0(name,'_expertsSamples.asc'),ncolumns=1)
    
    DDD = dominion
    
    for j in np.arange(N):
    
        s = int(sV[j])-1
        
        C[j]=max_entropy(incm[s],mid[s],incM[s],rA,rB)
        
        while((C[j]<DDD[0]) or (C[j]>DDD[1])):
        
            C[j]=max_entropy(incm[s],mid[s],incM[s],rA,rB)
    

    s = np

    if (not logPlot):
    
        C1=10**C
        DDD=10**DDD
    
    if (logPlot):
    
        C1=C

    C1 = C

    quan05 = np.quantile(C1,0.05)
    quan50 = np.quantile(C1,0.5)
    qmean = np.mean(C1)
    quan95 = np.quantile(C1,0.95)

    # write(C1,paste0(name,'SAMPLE.asc'),ncolumns=1)
    # print(quan05,quan50,qmean,quan95)

    """
    if (not dens):
    
        hist(C,col=col,density=dd,add=add,angle=angle,freq=F,xlab=list('DM Response',cex=1.4),ylab=list('Probability Density',cex=1.4),cex.axis=1.4)}
    
    if np.isinf(DDD[1]):
    
        PDF = density(C1)
    
    if not(np.isinf(DDD[1]):
    
        PDF = density(C1,from=DDD[0],to=DDD[1])

    if dens :
    
        if not(add):
        
            plot(PDF,main=name,col=col,lwd=2,xlab=list('DM Response',cex=1.4),cex.axis=1.4,ylab=list('Probability Density',cex=1.4))
            
        if add :
        
            lines(PDF,col=col,lwd=2)
    """
    
    return quan05,quan50,qmean,quan95,C
   
