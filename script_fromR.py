def createDATA1(DAT, j, W, N, logSCALE, dominion, ERF_flag):

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

    n = int(np.amax(DAT[:, 0]))
    nn = int(len(DAT[:, 0]) / n)

    incm = DAT[np.arange(n) * nn + j, 2]
    mid = DAT[np.arange(n) * nn + j, 3]
    incM = DAT[np.arange(n) * nn + j, 4]

    # print('incm',incm)
    # print('mid',mid)
    # print('incM',incM)

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

    # print(incm)
    # print(mid)
    # print(incM)

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


def generate_ERF(true_seed, SQ_array):

    # Calculation of the ERF weights of the experts, given seed questions
    # true values; arrays of 5th, 50th and 95th percentiles by the experts.

    import numpy as np
    import itertools

    Ne = SQ_array.shape[0]
    Nq = SQ_array.shape[2]

    pERF = np.zeros((Ne))
    p_single = np.zeros((Nq))

    for i in range(Ne):

        for j in range(Nq):

            p_single[j] = ERFweight(true_seed[j], SQ_array[i, 0, j],
                                    SQ_array[i, 1, j], SQ_array[i, 2, j])

        pERF[i] = np.mean(p_single)

    W = np.zeros((Ne, 5))
    W[:, 4] = pERF / np.sum(pERF)

    return W


def ERFweight(x, a, b, c):

    # ERF calculation, given a true answer and its elicitated percentiles.

    import numpy as np

    if (a == c):

        if (b == x):

            return (1)

        else:

            return (0)

    S = NewRap(a, b, c)
    a = S[0]
    c = S[1]
    A = np.minimum(np.maximum(0.95 * x, a), c)
    B = np.maximum(np.minimum(1.05 * x, c), a)
    p = ((A - c)**2 - (B - c)**2) / ((c - a) * (c - b))

    if (A < b):

        p = 1.0 - (B - c)**2 / ((c - a) * (c - b)) - (A - a)**2 / ((b - a) *
                                                                   (c - a))

        if (B < b):

            p = ((B - a)**2 - (A - a)**2) / ((b - a) * (c - a))

    return p


def NewRap(a, b, c):

    import numpy as np

    x0 = a - (c - a) / 6
    y0 = c + (c - a) / 6
    x = np.array([x0, y0])

    for i in range(5):

        M = InvJac(x[0], x[1], a, b, c)
        v = FunRap(x[0], x[1], a, b, c)

        x = x - M.dot(v)

    return x


def InvJac(x, y, a, b, c):

    import numpy as np

    A = 1.9 * x + 0.05 * y - 2 * a + 0.05 * b
    B = 0.05 * (x - b)
    C = 0.05 * (y - b)
    D = 1.9 * y + 0.05 * x - 2 * c + 0.05 * b

    M = np.zeros((2, 2))
    M[0, 0] = D
    M[1, 1] = A
    M[0, 1] = -B
    M[1, 0] = -C

    Mnew = M / (A * D - B * C)

    return Mnew


def FunRap(x, y, a, b, c):

    import numpy as np

    A1 = (a - x)**2 - 0.05 * (y - x) * (b - x)
    A2 = (y - c)**2 - 0.05 * (y - x) * (y - b)
    v = np.array([A1, A2])

    return v


def rtrian(a, b, c):

    import numpy as np

    # Random sample assuming a triangular distribution, given mode and 5th, 95th percentiles.

    if (a == c):

        R = b

    else:

        R = rtrian_inner(NewRap(a, b, c)[0], b, NewRap(a, b, c)[1])

    return R


def rtrian_inner(a, b, c):

    import numpy as np

    # Random sample assuming a triangular distribution, given mode and range.
    rng = np.random.default_rng()
    u = rng.random(1)

    if (u < ((b - a) / (c - a))):

        x = np.sqrt(u * (b - a) * (c - a)) + a

    else:

        x = c - np.sqrt((1 - u) * (c - a) * (c - b))

    return x
