def createSamples(DAT, j, W, N, logSCALE, domain, overshoot, ERF_flag):

    """Compute the quantiles and samples for question j from weights
    and answers

    Parameters
    ----------
    DAT : float numpy array [ n_experts * (n_SQ + n_TQ), n_pctl + 2 ]
        Numpy array with experts' answers to seed and target questions
    j : int
        Integer for question index
    W : float numpy array [ n_experts ]
        Numpy array with experts'weights
    N : int
        Integer for size of sample array
    logSCALE : int
        scale of question (0 for uni; 1 for log)
    domain : int list [ 2 ]
        domain for asnwer (domain[0] = minVal; domain[1] = maxVal)
    ERF_flag : int
        integer for method (1 for ERF; 2 for ERF mod; 0 for Cooke)

    Returns
    -------
    quan05 : float
        5%ile computed from array sample
    quan50 : float
        50%ile computed from array sample
    qmean : float
        mean computed from array sample
    quan95 : float
        95%ile computed from array sample
    C : float numpy array
        array with samples (size N)

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np

    n = int(np.amax(DAT[:, 0]))
    nn = int(len(DAT[:, 0]) / n)

    incm = DAT[np.arange(n) * nn + j, 2]
    mid = DAT[np.arange(n) * nn + j, 3]
    incM = DAT[np.arange(n) * nn + j, 4]

    if ERF_flag == 1:

        C = createSamplesERF_original(
            incm, mid, incM, W, N, logSCALE, domain
        )

    elif ERF_flag == 2:

        C = createSamplesERF(
            incm, mid, incM, W, N, logSCALE, domain
        )

    else:

        C = createSamplesUCA2(
            incm, mid, incM, W, N, logSCALE, domain, overshoot
        )

    return C


def max_entropy(incm, mid, incM, rA, rB):

    """Produces a random sample from a maximum entropy distribution

    Parameters
    ----------
    incm : float
        5th percentile of the distribution
    mid : float
        50th percentile of the distribution
    incM : float
        95th percentile of the distribution
    rA : float
        minimum value of the distribution
    rB : float
        maximum value of the distribution

    Returns
    -------
    y : float
        random sample

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np

    rng = np.random.default_rng()
    x = rng.random(1)
    y = mid + (incM - mid) * rng.random(1)

    if x > 0.95:

        y = incM + (rB - incM) * rng.random(1)

    if x < 0.5:

        y = incm + (mid - incm) * rng.random(1)

    if x < 0.05:

        y = rA + (incm - rA) * rng.random(1)

    return y


def sampleDISCR(P, N):

    """Produces an array of random samples from a discrete distribution

    Parameters
    ----------
    P : float numpy array [ n_experts ]
        discrete probability values
    N : int
        number of samples

    Returns
    -------
    a : float numpy array [ N ]
        random samples

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np

    n = len(P)
    PP = np.zeros(n)

    PP[0] = P[0]

    for i in np.arange(1, n - 1):

        PP[i] = PP[i - 1] + P[i]

    PP[n - 1] = 1

    rng = np.random.default_rng(seed=None)
    u = rng.random(N)
    a = np.zeros(N)

    for i in np.arange(N):

        B = u[i] - PP
        B = B[B <= 0]
        b = len(B)
        a[i] = n - b + 1

    return a


def createSamplesUCA2(incm, mid, incM, W, N, logSCALE, domain, overshoot):

    """Produces an array of random samples from weights and answers by using
    max. entropy distributions and linear pooling

    Parameters
    ----------
    incm : float numpy array [ n_experts ]
        5th percentiles of all the experts
    mid : float numpy array [ n_experts ]
        50th percentiles of all the experts
    incM : float numpy array [ n_experts ]
        95th percentiles of all the experts
    W : float numpy array [ n_experts ]
        experts' scores
    N : integer
        number of samples
    logSCALE : int
        scale of question (0 for uni; 1 for log)
    domain : int list [ 2 ]
        domain for asnwer (domain[0] = minVal; domain[1] = maxVal)

    Returns
    -------
    quan05 : float
        5%ile computed from array sample
    quan50 : float
        50%ile computed from array sample
    qmean : float
        mean computed from array sample
    quan95 : float
        95%ile computed from array sample
    C : float numpy array
        array with samples (size N)

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np

    W = W / np.sum(W)
    DDD = domain

    if logSCALE:

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
    rA = rA - overshoot * R
    rB = rB + overshoot * R

    sV = sampleDISCR(W, N)

    for j in np.arange(N):

        s = int(sV[j]) - 1

        C[j] = max_entropy(incm[s], mid[s], incM[s], rA, rB)

        while (C[j] < DDD[0]) or (C[j] > DDD[1]):

            C[j] = max_entropy(incm[s], mid[s], incM[s], rA, rB)

    if logSCALE:

        C1 = 10.0**C

    else:

        C1 = C

    return C1


def createSamplesERF_original(incm, mid, incM, W, N, logSCALE, domain):

    """Produces an array of random samples from weights and answers by using
    triangular distributions and quantile pooling

    Parameters
    ----------
    incm : float numpy array [ n_experts ]
        5th percentiles of all the experts
    mid : float numpy array [ n_experts ]
        modal values of all the experts
    incM : float numpy array [ n_experts ]
        95th percentiles of all the experts
    W : float numpy array [ n_experts ]
        experts' scores
    N : integer
        number of samples
    logSCALE : int
        scale of question (0 for uni; 1 for log)
    domain : int list [ 2 ]
        domain for asnwer (domain[0] = minVal; domain[1] = maxVal)

    Returns
    -------
    quan05 : float
        5%ile computed from array sample
    quan50 : float
        50%ile computed from array sample
    qmean : float
        mean computed from array sample
    quan95 : float
        95%ile computed from array sample
    C : float numpy array
        array with samples (size N)

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np
    from ERFweights import NewRap

    W = W / np.sum(W)

    DDD = domain

    if logSCALE:

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

    rng = np.random.default_rng(seed=None)
    u = rng.random(N)

    for i in range(Ne):

        Vp = NewRap(incm[i], mid[i], incM[i])
        a[i] = Vp[0]
        b[i] = mid[i]
        c[i] = Vp[1]

    for j in range(N):

        for i in range(Ne):

            if u[j] < ((b[i] - a[i]) / (c[i] - a[i])):

                P[i] = np.sqrt(u[j] * (b[i] - a[i]) * (c[i] - a[i])) + a[i]

            else:

                P[i] = c[i] - np.sqrt((1.0 - u[j]) * (c[i] - a[i]) *
                                      (c[i] - b[i]))

        C[j] = np.dot(P, W)

    if logSCALE:

        C1 = 10.0**C

    else:

        C1 = C

    return C1


def createSamplesERF(incm, mid, incM, W, N, logSCALE, domain):

    """Produces an array of random samples from weights and answers by using
    triangular distributions and linear pooling

    Parameters
    ----------
    incm : float numpy array [ n_experts ]
        5th percentiles of all the experts
    mid : float numpy array [ n_experts ]
        modal values of all the experts
    incM : float numpy array [ n_experts ]
        95th percentiles of all the experts
    W : float numpy array [ n_experts ]
        experts' scores
    N : integer
        number of samples
    logSCALE : int
        scale of question (0 for uni; 1 for log)
    domain : int list [ 2 ]
        domain for asnwer (domain[0] = minVal; domain[1] = maxVal)

    Returns
    -------
    quan05 : float
        5%ile computed from array sample
    quan50 : float
        50%ile computed from array sample
    qmean : float
        mean computed from array sample
    quan95 : float
        95%ile computed from array sample
    C : float numpy array
        array with samples (size N)

    This function is based on the R scripts
    written by A.Bevilacqua
    """

    import numpy as np
    from ERFweights import rtrian

    W = W / np.sum(W)

    DDD = domain

    if logSCALE:

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

        while (C[j] < DDD[0]) or (C[j] > DDD[1]):

            C[j] = rtrian(incm[s], mid[s], incM[s])

    if logSCALE:

        C1 = 10.0**C

    else:

        C1 = C

    return C1
