def ERFweights(true_seed, SQ_array):

    """Compute the weights with ERF formulation
    
    Parameters
    ----------
    true_seed : list
        Python list with realization of the seed questions
    SQ_array : numpy array
        Array with answers to seed questions
    
    Returns
    -------
    W : numpy array
        an array with weights
         
    This function is based on the R scripts
    written by A.Bevilacqua
    """
    
    import numpy as np
    import itertools

    Ne = SQ_array.shape[0]
    Nq = SQ_array.shape[2]

    pERF = np.zeros((Ne))
    p_single = np.zeros((Nq))

    for i in range(Ne):

        for j in range(Nq):

            p_single[j] = ERFcompute(true_seed[j], SQ_array[i, 0, j],
                                    SQ_array[i, 1, j], SQ_array[i, 2, j])

        pERF[i] = np.mean(p_single)

    W = np.zeros((Ne, 5))
    W[:, 4] = pERF / np.sum(pERF)

    return W,pERF


def ERFcompute(x, a, b, c):

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
