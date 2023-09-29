import numpy as np


def qthist(x, y, N=5, thresh=4, rng=[], density=True):
    '''
    Use a simple QuadTree approach to dynamically segment 2D
    data and compute a histogram (counts per bin). Since bin
    sizes are variable, by default the histogram returns the
    density (counts/area).

    QuadTree algorithm is implemented with `np.histogram2d`.

    Parameters
    ----------
    x, y : the 2 arrays of data to compute the histogram of
    N : int, optional, default = 5
        the number of levels to compute the QuadTree. Results
        in a maximum of [2**N, 2**N] bins
    thresh : int, optional, default = 4
        the number of points per bin to allow. Will keep
        segmenting bins until N levels is reached.
    range : the XY range to compute histogram over. Follows
        np.histogram2d convention, shape(2,2), optional.
        ``[[xmin, xmax], [ymin, ymax]]``. If not specified,
        `qthist` will use the XY limits of the data with a
        buffer of 1/4 the minimum bin size on each side.
    density : bool, optional, default = True
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin:
        ``num / len(x) / bin_area``.

    Returns
    -------
    num, xmin, xmax, ymin, ymax

    num : the array of number counts or densities per bin
    xmin,xmax,ymin,ymax : the left, right, bottom, top
        edges of each bin

    '''

    # start w/ 2x2 array of False leafs
    Mnext = np.empty((2**1,2**1),dtype='bool')*False

    # the 5 quantities to save in our Tree
    num = np.array([])
    xmin = np.array([])
    xmax = np.array([])
    ymin = np.array([])
    ymax = np.array([])

    # Step thru each level of the Tree
    for k in range(1, N+1):
        if len(rng) == 0:
            dx = (np.nanmax(x) - np.nanmin(x)) / (2**k)
            dy = (np.nanmax(y) - np.nanmin(y)) / (2**k)
            rng = [[np.nanmin(x)-dx/4, np.nanmax(x)+dx/4],
                   [np.nanmin(y)-dy/4, np.nanmax(y)+dy/4]]

        # lazily compute histogram of all data at this level
        H1, xedges1, yedges1 = np.histogram2d(x, y, range=rng, bins=2**k,)

        # any leafs at this level to pick, but NOT previously picked?
        if k<N:
            M1 = (H1 <= thresh)
        if k==N:
            # unless we on the last level, then pick the rest of the leafs
            M1 = ~Mnext

        Mprep = np.empty((2**(k+1),2**(k+1)),dtype='bool')*False

        # check leafs at this level
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                # up-scale the leaf-picking True/False to next level
                if k<N:
                    Mprep[(i*2):((i+1)*2),(j*2):((j+1)*2)] = M1[i,j] | Mnext[i,j]

                # if newly ready to pick, save 5 values
                if M1[i,j] & ~Mnext[i,j]:
                    num = np.append(num, H1[i,j])
                    xmin = np.append(xmin, xedges1[i])
                    xmax = np.append(xmax, xedges1[i+1])
                    ymin = np.append(ymin, yedges1[j])
                    ymax = np.append(ymax, yedges1[j+1])

        Mnext = Mprep

    if density:
        # following example from np.histogram:
        # result is the value of the probability *density* function at the bin,
        # normalized such that the *integral* over the range is 1
        num = num / ((ymax - ymin) * (xmax - xmin)) / num.sum()

    return num, xmin, xmax, ymin, ymax


def binned_statistic_qt(x, y, v=None, statistic='density', edges=None):
    '''
    Computed statistics inlike scipy's binned_statistic_2d, but for QT's, and assuming np.nan-friendly stats
    
    Parameters
    ----------
    x, y : the 2 arrays of new data to compute the histogram from
    v : array of values to compute statistic over. Needs to match 
        x and y
    edges : x and y edges of the Quad Tree, expanded as
        xmin,xmax,ymin,ymax = edges
    statistic : string or callable, optional
        The statistic to compute (default value is 'density')
        
        The following statistics are available:
            * 'density' : compute the count of points within each bin, then
                divide by the area of the bin in (x,y) space. This is useful
                for making a histogram-like plot from the quad tree.
            * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN. 
            * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN. 
            * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `v` array is not
                referenced. 
            * 'sum' : compute the sum of values for points within each bin. 
            * 'std' : compute the standard deviation within each bin.
                Empty bins will be represented by NaN.
            * 'min' : compute the minimum of values for points within each bin.
                Empty bins will be represented by NaN.
            * 'max' : compute the maximum of values for point within each bin.
                Empty bins will be represented by NaN.
            * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin. 

    Returns
    -------
    statistic : computed statistic within each bin
    '''
    
    #edges must be defined from QT... or we need to run QT to get them
    xmin,xmax,ymin,ymax = edges
    
    # check that statistic is callable
    known_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max', 'density']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError(f'invalid statistic {statistic!r}')
   
    result = np.zeros_like(xmin, dtype=np.float64)

    if statistic == 'density':
        for k in range(len(xmin)):
            result[k] = np.sum((x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k]))
        result = result / ((ymax - ymin) * (xmax - xmin)) / result.sum()
        
    elif statistic == 'count':
        for k in range(len(xmin)):
            result[k] = np.sum((x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k]))
            
    elif statistic == 'sum':
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nansum(v[b])
            
    elif statistic in {'mean', np.mean, np.nanmean}:
        result.fill(np.nan)
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nanmean(v[b])
            
    elif statistic in {'median', np.median, np.nanmedian}:
        result.fill(np.nan)
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nanmedian(v[b])
    elif statistic in {'min', np.min, np.nanmin}:
        result.fill(np.nan)
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nanmin(v[b])
    elif statistic in {'max', np.max, np.nanmax}:
        result.fill(np.nan)
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nanmax(v[b])
    elif statistic in {'std', np.std, np.nanstd}:
        result.fill(np.nan)
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = np.nanstd(v[b])

    # need to test callable statistic returns something that fits in result
    elif callable(statistic):
        for k in range(len(xmin)):
            b = (x >= xmin[k]) & (x < xmax[k]) & (y >= ymin[k]) & (y < ymax[k])
            result[k] = statistic(v[b])
        
    return result


def qtcount(x, y, xmin, xmax, ymin, ymax, density=True):
    '''
    given rectangular output ranges for cells/leafs from QThist
    count the occurence rate of NEW data in these cells
    
    NOTE: simply wraps `binned_statistic_qt`, but kept here for compatibility

    Parameters
    ----------
    x, y : the 2 arrays of new data to compute the histogram from
    xmin,xmax,ymin,ymax : the left, right, bottom, top
        edges of each bin, e.g. from previous ``qthist``.

    density : bool, optional, default = True
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin:
        ``num / len(x) / bin_area``.

    Returns
    -------
    num : the array of number counts or densities per bin
    
    '''
    num = binned_statistic_qt(x,y, statistic='count', edges=(xmin, xmax, ymin, ymax))

    return num
