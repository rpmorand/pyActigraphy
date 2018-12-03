import numpy as np
# import pandas as pd
import spm1d
import statsmodels.api as sm
# from ..io.base import BaseRaw
from joblib import Parallel, delayed


def _scotts_factor(n, d):
    return np.power(n, -1./(d+4))


def _silverman_factor(n, d):
    return np.power(n*(d+2.0)/4.0, -1./(d+4))


def _get_kernel_size(data, method, use_fwhm):

    # In order to mimic scipy.kde, the kernel covariance matrix is weighted
    # by the input data covariance matrix (N.B. in 1d, this simplifies...)
    data_std = data.std(ddof=1)

    # If Scotts' or Silverman's method is used, apply a factor
    # sqrt(8*log(2)) to convert sd to fwhm, if required
    sd2fwhm = np.sqrt(8*np.log(2))
    if use_fwhm:
        data_std *= sd2fwhm

    methods = ('scotts', 'silverman')
    if isinstance(method, str):
        if method not in methods:
            raise ValueError(
                '`method` must be "{}". You passed: "{}’"'.format(
                    '" or "'.join(methods),
                    method
                )
            )
        elif method == 'scotts':
            kernel_size = _scotts_factor(data.size, 1)*data_std
        elif method == 'silverman':
            kernel_size = _silverman_factor(data.size, 1)*data_std
    elif np.isscalar(method):
        kernel_size = method
    else:
        raise ValueError(
            '`method` must be "{}". You passed: "{}’"'.format(
                '" or "'.join(methods+['a scalar.']),
                method
            )
        )

    return kernel_size


class FLM():
    """ Class for Functional Linear Modelling"""

    def __init__(self, basis, sampling_freq, max_order=None):

        bases = ('fourier', 'spline', 'ssa', 'wavelet')
        if basis not in bases:
            raise ValueError(
                '`basis` must be "%s". You passed: "%s"' %
                ('" or "'.join(bases), basis)
            )

        self.__basis = basis
        self.__sampling_freq = sampling_freq
        self.__nsamples = None
        self.__max_order = max_order
        self.__basis_functions = None
        self.__beta = {}

    def __create_basis_functions(self, T):

        phi = []
        # Construct the fourier functions (cosine and sine)
        if self.__basis == 'fourier':
            # T = int(pd.Timedelta('24H')/pd.Timedelta(self.sampling_freq))
            omega = 2*np.pi / T
            t = np.linspace(0, T, T, endpoint=False)
            phi.append(np.cos(0 * t))
            for n in np.arange(1, self.max_order+1):
                phi.append(np.cos(n * omega * t))
                phi.append(np.sin(n * omega * t))

        self.basis_functions = phi

    def fit(self, raw, binarize=False, verbose=False):
        """Fit the actigraphy data using a basis function expansion.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be fitted.
        binarize: bool.
            If True, the data are binarized (i.e 0 or 1).
            Default is False.
        verbose : bool.
            If True, print the fit summary.
            Default is False.

        Returns
        -------
        y_est : ndarray
            Returns the functional form of the actigraphy data.
        """

        daily_avg = raw.average_daily_activity(
            binarize=binarize,
            freq=self.sampling_freq
        )
        self.__nsamples = daily_avg.index.size

        # Fourier
        if self.__basis == 'fourier':

            X = np.stack(self.basis_functions, axis=1)
            y = daily_avg.values
            model = sm.OLS(y, X)
            results = model.fit()

            if verbose:
                print(results.summary())

            self.__beta[raw.display_name] = results.params

        # Spline
        elif self.__basis == 'spline':

            from scipy.interpolate import splrep

            T = self.nsamples
            t = np.linspace(0, T, T, endpoint=False)
            k = 3 if self.max_order is None else self.max_order

            if verbose:
                print('Finding the {}-degree B-spline representation of'
                      'the input data'.format(k))

            self.__beta[raw.display_name] = list(
                splrep(t, daily_avg.values, k=k)
            )

    def evaluate(self, raw, r=10):
        """Evaluate the basis function expansion.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements used to create the basis functions.
        r : int
            Ratio between the number of points at which the basis functions are
            evaluated and the number of points at which the basis functions
            were fitted.
            Default is 10.
            N.B.: only valid for splines.

        Returns
        -------
        y_est : ndarray
            Returns the functional form of the actigraphy data.
        """

        if not self.beta:
            raise ValueError(
                'The basis function expansion parameters are empty.\n'
                'Please run the `self.fit` method first.'
            )

        # Fourier
        if self.__basis == 'fourier':
            X = np.stack(self.basis_functions, axis=1)
            y_est = np.dot(X, self.beta[raw.display_name])
            return y_est

        # Spline
        elif self.__basis == 'spline':
            from scipy.interpolate import splev
            T = self.nsamples
            t = np.linspace(0, T, r*T, endpoint=False, dtype=np.float)
            y_est = splev(t, tuple(self.beta[raw.display_name]))
            return y_est

    def fit_reader(
        self, reader,
        binarize=False, verbose_fit=False,
        n_jobs=1, prefer=None, verbose_parallel=0
    ):
        """Fit the actigraphy data contained in a reader using a basis function
        expansion.

        Parameters
        ----------
        reader : instance of RawReader
            Raw measurements to be fitted.
        binarize: bool.
            If True, the data are binarized (i.e 0 or 1).
            Default is False.
        verbose_fit : bool.
            If True, print the fit summary.
            Default is False.
        n_jobs: int
            Number of CPU to use for parallel fitting
        prefer: str
            Soft hint to choose the default backendself.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose_parallel: int
            Display a progress meter if set to a value > 0.
            Default is 0.

        """
        # avoid Parallel overhead if n_job == 1
        if(n_jobs == 1):
            for raw in reader.readers:
                self.fit(raw, binarize=binarize, verbose=verbose_fit)
        else:
            def parallel_fitting(raw, binarize, verbose_fit):
                self.fit(raw, binarize, verbose_fit)
            Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose_parallel)(
                delayed(parallel_fitting)(
                    raw=raw, binarize=binarize, verbose_fit=verbose_fit
                ) for raw in reader.readers
            )

    def evaluate_reader(
        self, reader,
        r=10,
        n_jobs=1, prefer=None, verbose_parallel=0
    ):
        """Evaluate the basis function expansion made on actigraphy data
        contained in a reader.

        Parameters
        ----------
        reader : instance of RawReader
            Raw measurements to be fitted.
        r : int
            Ratio between the number of points at which the basis functions are
            evaluated and the number of points at which the basis functions
            were fitted.
            Default is 10.
            N.B.: only valid for splines.
        n_jobs: int
            Number of CPU to use for parallel fitting
        prefer: str
            Soft hint to choose the default backendself.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose_parallel: int
            Display a progress meter if set to a value > 0.
            Default is 0.

        Returns
        -------
        y_est : ndarray
            Returns an array with functional forms of the actigraphy data.
        """
        # avoid Parallel overhead if n_job == 1
        if(n_jobs == 1):
            return dict([
                (raw.display_name, self.evaluate(raw, r))
                for raw in reader.readers
            ])
        else:
            def parallel_evaluating(raw, r):
                return (raw.display_name, self.evaluate(raw, r))
            return dict(Parallel(
                n_jobs=n_jobs, prefer=prefer, verbose=verbose_parallel
            )(
                delayed(parallel_evaluating)(
                    raw=raw, r=r
                ) for raw in reader.readers
            ))

    def smooth(self, raw, binarize=False, method='scotts', verbose=False):
        """Smooth the actigraphy data using a gaussian kernel.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be smoothed.
        binarize: bool.
            If True, the data are binarized (i.e 0 or 1).
            Default is False.
        method: str, float.
            Method to calculate the full-width at half-maximum of the gaussian
            kernel. Available methods are `scotts`, `silverman`. Method can be
            a scalar value too.
            Default is `scotts`.
        verbose: bool.
            If True, print the kernel size used to smooth the data.
            Default is False.

        Returns
        -------
        y_est : ndarray
            Returns the smoothed form of the actigraphy data.
        """

        daily_avg = raw.average_daily_activity(
            binarize=binarize,
            freq=self.sampling_freq
        )

        # Calculate optimal kernel size
        fwhm = _get_kernel_size(daily_avg.values, method=method, use_fwhm=True)

        if verbose:
            print('Kernel size used to smooth the data: {}'.format(fwhm))

        return spm1d.util.smooth(
            daily_avg,
            fwhm=fwhm
        )

    @property
    def sampling_freq(self):
        """The sampling frequency of the basis functions."""
        return self.__sampling_freq

    @sampling_freq.setter
    def sampling_freq(self, value):
        self.__sampling_freq = value

    @property
    def nsamples(self):
        """The number of sample points for the basis functions."""
        return self.__nsamples

    @property
    def max_order(self):
        """The maximal number of basis functions."""
        return self.__max_order

    @property
    def basis_functions(self):
        """The basis functions."""
        if self.__basis_functions is None:
            print("Create first the basis functions: {}".format(
                    self.__basis
                )
            )
            self.__create_basis_functions(self.nsamples)
        return self.__basis_functions

    @basis_functions.setter
    def basis_functions(self, value):
        self.__basis_functions = value

    @property
    def beta(self):
        """The scalar parameters of the basis function expansion."""
        return self.__beta
