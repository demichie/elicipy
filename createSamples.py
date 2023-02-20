def createSamples(DAT, j, W, N, logSCALE, dominion, ERF_flag):

    # input:
    # DAT risposte a seed e target questions
    # W vettore dei pesi di Cooke per target
    # j indice della domanda
    # N non si cambia mai
    # logscale si potrebbe anche prendere dal file DAT
    # dominion dice che campioni eliminare

    import numpy as np

    n = int(np.amax(DAT[:, 0]))
    nn = int(len(DAT[:, 0]) / n)

    incm = DAT[np.arange(n) * nn + j, 2]
    mid = DAT[np.arange(n) * nn + j, 3]
    incM = DAT[np.arange(n) * nn + j, 4]

    if ERF_flag ==1:

        quan05, quan50, qmean, quan95, C = createSamplesERF_original(
            incm, mid, incM, W, N, logSCALE, dominion)
            
    elif ERF_flag ==2:

        quan05, quan50, qmean, quan95, C = createSamplesERF(
            incm, mid, incM, W, N, logSCALE, dominion)

    else:

        quan05, quan50, qmean, quan95, C = createSamplesUCA2(
            incm, mid, incM, W, N, logSCALE, dominion)

    return quan05, quan50, qmean, quan95, C


def max_entropy(incm, mid, incM, rA, rB):

    import numpy as np

    rng = np.random.default_rng()
    x = rng.random(1)
    y = mid + (incM - mid) * rng.random(1)

    if (x > 0.95):

        y = incM + (rB - incM) * rng.random(1)

    if (x < 0.5):

        y = incm + (mid - incm) * rng.random(1)

    if (x < 0.05):

        y = rA + (incm - rA) * rng.random(1)

    return y


def sampleDISCR(P, N):

    import numpy as np

    n = len(P)
    PP = np.zeros(n)

    PP[0] = P[0]

    for i in np.arange(1, n - 1):

        PP[i] = PP[i - 1] + P[i]

    PP[n - 1] = 1
    # print(PP)

    rng = np.random.default_rng(12345)
    u = rng.random(N)
    a = np.zeros(N)

    for i in np.arange(N):

        B = u[i] - PP
        B = B[B <= 0]
        b = len(B)
        a[i] = n - b + 1

        # print(c(u[i],a[i]))

    return a


def createSamplesUCA2(incm, mid, incM, W, N, logSCALE, dominion):

    import numpy as np

    W = W / np.sum(W)
    DDD = dominion

    if (logSCALE):

        VV = incm > 0
        incm = np.log10(incm[VV])
        incM = np.log10(incM[VV])
        mid = np.log10(mid[VV])
        W = W[VV]
        W = W / np.sum(W)
        if DDD[0] > 0:

            DDD[0] = np.log10(DDD[0])

        else:

            DDD[0] = -np.inf

        DDD[1] = np.log10(DDD[1])

    C = np.zeros(N)

    rA = np.amin(incm[W > 0])
    rB = np.amax(incM[W > 0])
    R = rB - rA
    rA = rA - R / 10.0
    rB = rB + R / 10.0

    sV = sampleDISCR(W, N)

    for j in np.arange(N):

        s = int(sV[j]) - 1

        C[j] = max_entropy(incm[s], mid[s], incM[s], rA, rB)

        while ((C[j] < DDD[0]) or (C[j] > DDD[1])):

            C[j] = max_entropy(incm[s], mid[s], incM[s], rA, rB)

    s = np

    if (logSCALE):

        C1 = 10.0**C

    else:

        C1 = C

    quan05 = np.quantile(C1, 0.05)
    quan50 = np.quantile(C1, 0.5)
    qmean = np.mean(C1)
    quan95 = np.quantile(C1, 0.95)

    return quan05, quan50, qmean, quan95, C1


#sampler quantile averaging

def createSamplesERF_original(incm,mid,incM,W,N, logSCALE, dominion):

    import numpy as np
    from ERFweights import NewRap

    W = W / np.sum(W)

    DDD = dominion

    if (logSCALE):

        VV = incm > 0
        incm = np.log10(incm[VV])
        incM = np.log10(incM[VV])
        mid = np.log10(mid[VV])
        W = W[VV]
        W = W / np.sum(W)
        if DDD[0] > 0:

            DDD[0] = np.log10(DDD[0])

        else:

            DDD[0] = -np.inf

        DDD[1] = np.log(DDD[1])

    C = np.zeros(N)
    C = np.zeros(N)

    Ne = len(incm)

    P = np.zeros_like(incm)
    a = np.zeros_like(incm)
    b = np.zeros_like(incm)
    c = np.zeros_like(incm)

    rng = np.random.default_rng(12345)
    u = rng.random(N)
    

    for i in range(Ne):
    
        Vp = NewRap(incm[i],mid[i],incM[i])
        a[i]=Vp[0]
        b[i]=mid[i]
        c[i]=Vp[1]

 

    for j in range(N):
        
        for i in range(Ne):
        
            if (u[j]<((b[i]-a[i])/(c[i]-a[i]))):
            
                P[i] = np.sqrt(u[j]*(b[i]-a[i])*(c[i]-a[i]))+a[i]

            else:
            
                P[i] = c[i] - np.sqrt((1.0-u[j])*(c[i]-a[i])*(c[i]-b[i]))

        C[j] = np.dot(P,W)

    if (logSCALE):

        C1 = 10.0**C

    else:

        C1 = C

    quan05 = np.quantile(C1, 0.05)
    quan50 = np.quantile(C1, 0.5)
    qmean = np.mean(C1)
    quan95 = np.quantile(C1, 0.95)

    return quan05, quan50, qmean, quan95, C1

def createSamplesERF(incm, mid, incM, W, N, logSCALE, dominion):

    import numpy as np

    W = W / np.sum(W)

    DDD = dominion

    if (logSCALE):

        VV = incm > 0
        incm = np.log10(incm[VV])
        incM = np.log10(incM[VV])
        mid = np.log10(mid[VV])
        W = W[VV]
        W = W / np.sum(W)
        if DDD[0] > 0:

            DDD[0] = np.log10(DDD[0])

        else:

            DDD[0] = -np.inf

        DDD[1] = np.log(DDD[1])

    C = np.zeros(N)

    sV = sampleDISCR(W, N)

    for j in np.arange(N):

        s = int(sV[j]) - 1

        C[j] = rtrian(incm[s], mid[s], incM[s])

        while ((C[j] < DDD[0]) or (C[j] > DDD[1])):

            C[j] = rtrian(incm[s], mid[s], incM[s])

    if (logSCALE):

        C1 = 10.0**C

    else:

        C1 = C

    quan05 = np.quantile(C1, 0.05)
    quan50 = np.quantile(C1, 0.5)
    qmean = np.mean(C1)
    quan95 = np.quantile(C1, 0.95)

    return quan05, quan50, qmean, quan95, C1

