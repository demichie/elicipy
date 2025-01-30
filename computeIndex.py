def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    from:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """

    import numpy as np

    values = np.ravel(values)
    quantiles = np.ravel(quantiles)
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.ravel(sample_weight)

    values[np.isnan(values)] = 0
    sample_weight[np.isnan(values)] = 0

    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def calculate_index(TQ_array, weight, background_measure):
    import numpy as np

    Nq = TQ_array.shape[2]
    E = TQ_array.shape[0]

    indexTot = np.zeros((Nq, E, E))
    indexMean = np.zeros(Nq)
    indexStd = np.zeros(Nq)
    indexQuantiles = np.zeros((Nq, 3))

    weightTable = np.outer(weight, weight)

    # loop over target questions
    for i in np.arange(Nq):

        # loop over experts
        # index j for first expert
        for j in np.arange(E):

            if background_measure[i] == "uni":

                mj = TQ_array[j, 0, i]
                Mj = TQ_array[j, 2, i]

            elif background_measure[i] == "log":

                mj = np.log(TQ_array[j, 0, i])
                Mj = np.log(TQ_array[j, 2, i])

            # loop over experts
            # index h for second expert
            for h in np.arange(E):

                # compute the index for different experts only
                if h != j:

                    if background_measure[i] == "uni":

                        mh = TQ_array[h, 0, i]
                        Mh = TQ_array[h, 2, i]

                    elif background_measure[i] == "log":

                        mh = np.log(TQ_array[h, 0, i])
                        Mh = np.log(TQ_array[h, 2, i])

                    m_un = np.minimum(mj, mh)
                    M_un = np.maximum(Mj, Mh)

                    m_in = np.maximum(mj, mh)
                    M_in = np.minimum(Mj, Mh)

                    indexTot[i, j, h] = (M_in - m_in) / (M_un - m_un)

                else:

                    indexTot[i, j, h] = np.nan

        indexQuantiles[i, :] = weighted_quantile(indexTot[i, :, :],
                                                 [0.05, 0.5, 0.95],
                                                 sample_weight=weightTable)

        ma = np.ma.MaskedArray(indexTot[i, :, :],
                               mask=np.isnan(indexTot[i, :, :]))
        indexMean[i] = np.average(ma, weights=weightTable)
        variance = np.average((ma-indexMean[i])**2, weights=weightTable)
        indexStd[i] = np.sqrt(variance)

    return indexMean, indexStd, indexQuantiles
