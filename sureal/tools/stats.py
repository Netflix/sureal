import numpy as np
import scipy
import scipy.signal
from .inverse import inversefunc
import warnings

# import multiprocessing
# pool = multiprocessing.Pool()

from .misc import parallel_map

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def vectorized_gaussian(xs, locs, scales):
    # f = lambda x, scale: stats.norm.pdf(x, scale=scale)
    # ff = np.vectorize(f)
    # return ff(xs - locs, scales)

    return 1.0 / np.sqrt(2 * np.pi) / scales * np.exp(- (xs - locs)**2 / (2* scales**2))


def vectorized_logistic(xs, locs, scales):
    # f = lambda x, scale: stats.logistic.pdf(x, scale=scale)
    # ff = np.vectorize(f)
    # return ff(xs - locs, scales)

    return 1.0 / 4.0 / scales / np.cosh((xs - locs) / 2.0 / scales)**2


def sech(x):
    return 2. / ( np.exp(x) + np.exp(-x) )


def vectorized_convolution_of_two_logistics(xs, locs1, scales1, locs2, scales2):

    f = lambda x, loc1, scale1, loc2, scale2: \
        ConvolveTwoPdf(
            lambda x: 1.0 / 4.0 / scale1 * sech(x / 2.0 / scale1)**2,
            lambda x: 1.0 / 4.0 / scale2 * sech(x / 2.0 / scale2)**2,

            # lambda x: 1.0 / 4.0 / scale1 / np.cosh(x / 2.0 / scale1)**2,
            # lambda x: 1.0 / 4.0 / scale2 / np.cosh(x / 2.0 / scale2)**2,

            # lambda x: 1.0 / np.sqrt(2 * np.pi * (scale1**2) * (np.pi**2 / 3.)) * np.exp(- x**2 / (2* (scale1**2) * (np.pi**2 / 3.))), # test gaussian
            # lambda x: 1.0 / np.sqrt(2 * np.pi * (scale2**2) * (np.pi**2 / 3.)) * np.exp(- x**2 / (2* (scale2**2) * (np.pi**2 / 3.))), # test gaussian

            f_truncation=1e-12,
            g_truncation=1e-12,
            delta=3.0e-3,
        ).pdf(x - loc1 - loc2)

    # # === way 1: parallel_map (each job too small, bottlenecked by passing context) ===
    # f2 = lambda x: f(*x)
    # xshape = xs.shape
    # assert xshape == locs1.shape == scales1.shape == locs2.shape == scales2.shape
    # res = parallel_map(f2, zip(xs.ravel(), locs1.ravel(), scales1.ravel(), locs2.ravel(), scales2.ravel()), pause_sec=None)
    # return np.reshape(res, xshape)

    # # === way 2: vectorize (sequential execution) ===
    # ff = np.vectorize(f)
    # return ff(xs, locs1, scales1, locs2, scales2)

    # === way 3: parallel map combined with vectorize (best speed) ===
    ff = np.vectorize(f)
    ff2 = lambda x: ff(*x)
    xshape = xs.shape
    assert xshape == locs1.shape == scales1.shape == locs2.shape == scales2.shape
    with np.errstate(over='ignore'):
        res = parallel_map(ff2, zip(xs, locs1, scales1, locs2, scales2), pause_sec=None)
    return np.array(res)

    # === test: test one gaussian ===
    # return 1.0 / np.sqrt(2 * np.pi * (scales1**2 + scales2**2) * (np.pi**2 / 3.)) * np.exp(- (xs - locs1 - locs2)**2 / (2* (scales1**2 + scales2**2) * (np.pi**2 / 3.)))


def convolution_of_two_uniforms(x, loc1, s1, loc2, s2):
    """
    >>> convolution_of_two_uniforms(-2, 0, 1, 0, 2)
    0.0
    >>> convolution_of_two_uniforms(-1.5, 0, 1, 0, 2)
    0.0
    >>> convolution_of_two_uniforms(-1.49, 0, 1, 0, 2)
    0.0050000000000000044
    >>> convolution_of_two_uniforms(-0.51, 0, 1, 0, 2)
    0.495
    >>> convolution_of_two_uniforms(-0.49, 0, 1, 0, 2)
    0.5
    >>> convolution_of_two_uniforms(0, 0, 1, 0, 2)
    0.5
    >>> convolution_of_two_uniforms(0.49, 0, 1, 0, 2)
    0.5
    >>> convolution_of_two_uniforms(0.51, 0, 1, 0, 2)
    0.495
    >>> convolution_of_two_uniforms(1.49, 0, 1, 0, 2)
    0.0050000000000000044
    >>> convolution_of_two_uniforms(1.5, 0, 1, 0, 2)
    0.0
    >>> convolution_of_two_uniforms(2, 0, 1, 0, 2)
    0.0
    """
    z = x - loc1 - loc2
    d = abs(s1 - s2)
    s = s1 + s2
    h = 2. / (d + s)

    if - s/2. <= z < - d/2.:
        x0, y0 = - s / 2., 0
        x1, y1 = -d / 2., h
        return (y1 - y0) / (x1 - x0) * (x - x0) + y0
    elif -d/2. <= z < d/2.:
        return h
    elif d/2. <= z < s/2.:
        x0, y0 = s / 2., 0
        x1, y1 = d / 2., h
        return (y1 - y0) / (x1 - x0) * (x - x0) + y0
    else:
        return 0.


def vectorized_convolution_of_two_uniforms(xs, locs1, scales1, locs2, scales2):
    return np.vectorize(convolution_of_two_uniforms)(xs, locs1, scales1, locs2, scales2)


class ConvolveTwoPdf(object):
    """
    Generate a object of probability density function, which is the convolution of two
    valid probability density function. The resulting object is able to evaluate its
    probability density at any real value.
    """

    def __init__(self, f, g, delta=1e-2, f_truncation=1e-5, g_truncation=1e-5):
        self.f = f
        self.g = g
        self.delta = delta
        self.f_truncation=f_truncation
        self.g_truncation=g_truncation

        self.model = None

    def pdf(self, x):
        if self.model is None:
            self._get_model()

        return self._pdf(x)

    def _get_model(self):
        inv_f = inversefunc(self.f, self.f_truncation)
        inv_g = inversefunc(self.g, self.g_truncation)
        assert inv_f > 0
        assert inv_g > 0
        reach = max(inv_f, inv_g)
        big_grid = np.arange(-reach, reach, self.delta)
        pmf_f = self.f(big_grid) * self.delta
        pmf_f = (pmf_f + np.hstack([pmf_f[1:], pmf_f[-1]])) / 2.  # trapezoidal rule for better accuracy
        pmf_g = self.g(big_grid) * self.delta
        pmf_g = (pmf_g + np.hstack([pmf_g[1:], pmf_g[-1]])) / 2.  # trapezoidal rule for better accuracy
        conv_pmf = scipy.signal.fftconvolve(pmf_f, pmf_g, 'same')

        # try:
        #     np.testing.assert_almost_equal(sum(conv_pmf), 1, decimal=3)
        # except AssertionError:
        #     warnings.warn('expect sum(conv_pmf) close to 1.0 but is {}'.format(sum(conv_pmf)))

        conv_pdf = conv_pmf / self.delta

        self.model = {
            'grid': big_grid,
            'pdf': conv_pdf,
        }

    def _pdf(self, x):
        assert self.model is not None
        return np.interp(x, self.model['grid'], self.model['pdf'],
                         left=self.f_truncation*self.g_truncation,
                         right=self.f_truncation*self.g_truncation)


def get_cdf(x, bins=100):
    x = np.array(x)
    counts, bin_edges = np.histogram(x, bins=bins)
    cdf = np.cumsum(counts)
    cdf = cdf / float(cdf[-1]) # normalize
    bin_edges = bin_edges[1:] # make size
    return cdf, bin_edges


def get_pdf(data, bins=20, density=True):
    pdf, bin_edges = np.histogram(data, density=density, bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    return pdf, bin_centres


def histc(series):
    dd = dict()
    for x in series:
        if np.isnan(x):
            continue
        dd.setdefault(x, 0)
        dd[x] += 1
    return dd