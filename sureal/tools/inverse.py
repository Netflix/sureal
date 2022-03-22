__author__ = "Alvaro Sanchez-Gonzalez"
__license__ = "MIT"

import numpy as np
import warnings

from scipy.optimize import minimize_scalar

__all__ = ['inversefunc']


def inversefunc(func,
                y_values=None,
                domain=None,
                image=None,
                open_domain=None,
                args=(),
                accuracy=2):
    r"""Obtain the inverse of a function.

    Returns the numerical inverse of the function `f`. It may return a callable
    that can be used to calculate the inverse, or the inverse of certain points
    depending on the `y_values` argument.
    In order for the numerical inverse to exist in its domain, the
    input function must have, continuous, strictly monotonic behavior i.e. be
    purely decreasing or purely increasing in that domain. By default the
    domain interval spans all the real numbers, however it can be restricted
    with the `domain` and `open_domain` arguments. The image of the function
    in the interval may be provided, for cases where the function is non
    continuous right at the end of an open interval.

    Parameters
    ----------
    func : callable
        Callable representing the function to be inverted, able to take a
        ndarray or an scalar and return an object of the same kind with the
        evaluation of the function.  If `func` takes many arguments, it is
        inverted along the axis corresponding to the first argument.
        The function must not diverge and have a continuous strictly monotonic
        behavior in the chosen interval.
    y_values : float, ndarray, optional
        Values for which calculate the inverse function. If set to None, then
        a callable that can be used to calculate the inverse of values is
        returned. Default None.
    domain : float, ndarray, optional
        Boundaries of the domain (`domain[0]`, `domain[1]`).
        `domain[1]` must be larger than `domain[0]`.
        None values are assumed to be no boundary in that direction.
        A single scalar value will set it to [`domain`, None].
        Default None (-Inf, Inf).
    open_domain : bool, ndarray, optional
        Whether the domain is an open interval at each of the ends.
        A single scalar boolean will set it to [`open_domain`, `open_domain`].
        Default None [False, False].
    image : float, ndarray, optional
        Image of the function in the domain (`image[0]`, `image[1]`).
        `image[1]` must be larger than `image[0]`.
        None values are assumed to be no boundary in that direction.
        Default None, this is (-Inf, Inf) if domain is None, or the limits
        set by func(domain[0]) and func(domain[1]).
    args : tuple, optional
        Extra arguments to pass to `func`. Default ().
    accuracy : int, optional
        Number of digits for the desired accuracy. It will give a warning
        if the accuracy is worse than this.
        Default 2.

    Returns
    -------
    callable or ndarray
        Inverse function of `func`. It can take scalars or ndarrays, and return
        objects of the same kind with the calculated inverse values.

    Notes
    -----

    .. versionadded:: 0.19.0

    Examples
    --------
    >>> import numpy as np
    >>> cube = (lambda x: x**3)
    >>> invcube = inversefunc(cube)
    >>> invcube(27) # Should give 3
    array(3.00000001)
    >>> invsquare = inversefunc(np.power, args=(2), domain=0)
    >>> invsquare([4, 16, 64]) # Should give [2, 4, 8]
    array([2., 4., 8.])
    >>> inversefunc(np.log10, y_values=-2, # Should give 0.01
    ...             domain=0, open_domain=True)
    array(0.01)
    >>> inversefunc(np.cos, y_values=[1, 0, -1], # Should give [0, pi / 2, pi]
    ...             domain=[0, np.pi])
    array([0.        , 1.57079632, 3.14159265])
    >>> invtan = inversefunc(np.tan,
    ...                      domain=[-np.pi / 2, np.pi / 2],
    ...                      open_domain=True)
    >>> invtan([1, 0, -1]) # Should give [pi / 4, 0, -pi / 4]
    array([ 7.85398163e-01,  1.29246971e-26, -7.85398163e-01])

    """

    domain, image, open_domain, args = _normparams_inversefunc(domain,
                                                               image,
                                                               open_domain,
                                                               args)

    ymin, ymax = image
    xmin, xmax = domain
    xmin_open, xmax_open = open_domain

    # Calculating if the function is increasing or decreasing, using ref points
    # anywhere in the valid range (Function has to be strictly monotonic)
    ref1, ref2 = _get_valid_refpoints(xmin, xmax)
    trend = np.sign(func(ref2, *args) - func(ref1, *args))

    if trend == 0:
        raise ValueError("Function is not strictly monotonic")

    # Calculating the image by default
    if ymin is None:
        ymin = _auto_ymin(func, args, xmin, xmax, trend)
    if ymax is None:
        ymax = _auto_ymax(func, args, xmin, xmax, trend)

    # Creating bounded function
    def bounded_f(x):
        if xmin is not None and (x < xmin or (x == xmin and xmin_open)):
                val = -1 * np.inf * trend
        elif xmax is not None and (x > xmax or (x == xmax and xmax_open)):
                val = np.inf * trend
        else:
            val = func(x, *args)
        return val

    min_kwargs = {}
    min_kwargs['bracket'] = (ref1, ref2)
    min_kwargs['tol'] = 1.48e-08
    min_kwargs['method'] = 'Brent'

    def inv(yin):
        yin = np.asarray(yin, dtype=np.float64)
        shapein = yin.shape
        yin = yin.flatten()
        if ymin is not None:
            if (xmin_open and trend == 1) or (xmax_open and trend == -1):
                mask = yin <= ymin
            else:
                mask = yin < ymin
            if yin[mask].size > 0:
                raise ValueError("Requested values %s lower than the"
                                 " lower limit %g of the image" %
                                 (yin[mask], ymin))
        if ymax is not None:
            if (xmax_open and trend == 1) or (xmin_open and trend == -1):
                mask = yin >= ymax
            else:
                mask = yin > ymax
            if yin[mask].size > 0:
                raise ValueError("Requested values %s higher than the"
                                 " higher limit %g of the image" %
                                 (yin[mask], ymax))

        results = yin.copy() * np.nan
        resultsmask = np.zeros(yin.shape, dtype=bool)

        for j in range(yin.size):
            if xmax is not None:
                if bounded_f(xmax) == yin[j]:
                    results[j] = xmax
                    resultsmask[j] = True
                    continue
            if xmin is not None:
                if bounded_f(xmin) == yin[j]:
                    results[j] = xmin
                    resultsmask[j] = True
                    continue

            optimizer = (lambda x, j=j,
                         bounded_f=bounded_f: (((bounded_f(x) - yin[j]))**2))
            try:
                with warnings.catch_warnings(record=True):
                    result = minimize_scalar(optimizer, **min_kwargs)
                results[j] = result.x
                resultsmask[j] = result.success
            except:
                resultsmask[j] = False
        if any(~resultsmask):
            warnings.warn("Trouble calculating inverse for values: "
                          "%s" % str(yin[~resultsmask]), RuntimeWarning)

        # try:
        #     np.testing.assert_array_almost_equal(yin, func(results, *args),
        #                                          decimal=accuracy)
        # except AssertionError:
        #     warnings.warn("Results obtained with less than %g "
        #                   "decimal digits of accuracy"
        #                   % accuracy, RuntimeWarning)

        return results.reshape(shapein)

    if y_values is None:
        return inv
    else:
        return inv(y_values)


def _normparams_inversefunc(domain, image, open_domain, args):

    if not isinstance(args, tuple):
        args = (args,)

    error_domain = ("domain must be a single scalar, or a have two "
                    "elements [xmin, xmax]. Set None, to leave it "
                    "unlimited on one side.")
    if domain is None:
        domain = (None, None)
    else:
        domain = np.asarray(domain)
        if domain.ndim == 0:
            domain = (float(domain), None)
        elif domain.ndim == 1 and domain.size != 2:
            raise ValueError(error_domain)
        elif domain.ndim > 1:
            raise ValueError(error_domain)
        else:
            domain = ((float(domain[0]) if domain[0] is not None else None),
                      (float(domain[1]) if domain[1] is not None else None))

    error_open_domain = ("open_domain must be a single scalar, or a have two "
                         "bool elements [open_xmin, open_xmax].")
    if open_domain is None:
        open_domain = (False, False)
    else:
        open_domain = np.asarray(open_domain)
        if open_domain.ndim == 0:
            open_domain = (bool(open_domain), bool(open_domain))
        elif open_domain.ndim == 1 and open_domain.size != 2:
            raise ValueError(error_open_domain)
        elif open_domain.ndim > 1:
            raise ValueError(error_open_domain)
        else:
            open_domain = (bool(open_domain[0]), bool(open_domain[1]))

    error_image = ("image must be a single scalar, or a have two "
                   "bool elements [ymin, ymax].")
    if image is None:
        image = (None, None)
    else:
        image = np.asarray(image)
        if image.ndim != 1 or image.size != 2:
            raise ValueError(error_image)
        else:
            image = ((float(image[0]) if image[0] is not None else None),
                     (float(image[1]) if image[1] is not None else None))

    if domain[0] is not None and domain[1] is not None:
        if domain[0] >= domain[1]:
            raise ValueError("domain[0] min must be less than domain[1]")

    if image[0] is not None and image[1] is not None:
        if image[0] >= image[1]:
            raise ValueError("image[0] min must be less than image[1]")

    return domain, image, open_domain, args


def _get_valid_refpoints(xmin, xmax):
    if xmin is not None and xmax is not None:
        d = xmax - xmin
        ref1 = xmin + d / 4.
        ref2 = xmax - d / 4.
    elif xmin is not None:
        ref1 = xmin + 1.
        ref2 = xmin + 2.
    elif xmax is not None:
        ref1 = xmax - 2.
        ref2 = xmax - 1.
    else:
        ref1 = 0.
        ref2 = 1.
    return ref1, ref2


def _auto_ymin(func, args, xmin, xmax, trend):
    ymin = None
    if ((xmin is not None and trend == 1) or
       (xmax is not None and trend == -1)):
        try:
            with warnings.catch_warnings(record=True):
                ymin = func(xmin, *args) if trend == 1 else func(xmax, *args)
        except:
            raise ValueError("Cannot automatically calculate the lower limit "
                             "of the image please inclue it as a parameter")
    return ymin


def _auto_ymax(func, args, xmin, xmax, trend):
    ymax = None
    if ((xmax is not None and trend == 1) or
       (xmin is not None and trend == -1)):
        try:
            with warnings.catch_warnings(record=True):
                ymax = func(xmax, *args) if trend == 1 else func(xmin, *args)
        except:
            raise ValueError("Cannot automatically calculate the upper limit "
                             "of the image please include it as a parameter")
    return ymax

