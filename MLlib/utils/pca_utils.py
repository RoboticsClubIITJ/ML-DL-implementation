import numpy as np
from math import log, gamma


''' Gammaln function of scipy.special library'''


def gammaln(a):
    b = []
    for i in np.nditer(a):
        b.append(gamma(i))
    b = np.array(b).reshape(a.shape)
    b = np.log(np.absolute(b))
    return b


def assess_dimension(spectrum, rank, n_samples):
    """
    Compute the log-likelihood of a rank 'rank' dataset.
    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum 'spectrum'.
    """
    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError("The tested rank should be in [1, n_features - 1]")
    eps = 1e-15

    if spectrum[rank - 1] < eps:
        return -np.inf
    pu = -rank * log(2.)
    for i in range(1, rank + 1):
        pu += (gammaln((n_features - i + 1) / 2.) - log(np.pi) *
               (n_features - i + 1) / 2.)
    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.
    v = max(eps, np.sum(spectrum[rank:]) / (n_features - rank))
    pv = -np.log(v) * n_samples * (n_features - rank) / 2.
    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank) / 2.
    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)
    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.
    return ll


def infer_dimension(spectrum, n_samples):
    """
    Infers the dimension of a dataset with a given spectrum.
    The returned value will be in [1, n_features - 1].
    """
    ll = np.empty_like(spectrum)
    ll[0] = -np.inf  # we don't want the n_components to be 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = assess_dimension(spectrum, rank, n_samples)
    return ll.argmax()


class PCA_utils():
    def get_covariance(self):
        """Compute data covariance with the generative model.
        ``cov = components.T * S**2 * components + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        """
        if self.fitted is False:
            raise ValueError("The model should be fitted first.")
        components = self.components
        exp_var = self.explained_variances
        if self.whiten:
            components = components * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance, 0.)
        cov = np.dot(components.T * exp_var_diff, components)
        cov.flat[::len(cov) + 1] += self.noise_variance
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.
        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.
        """
        if self.fitted is False:
            raise ValueError("The model should be fitted first.")
        n_features = self.components.shape[1]
        # handle corner cases
        if self.n_components == 0:
            return np.eye(n_features) / self.noise_variance
        if self.n_components == n_features:
            return np.linalg.inv(self.get_covariance())
        # Get precision using matrix inversion lemma
        components = self.components
        exp_var = self.explained_variances
        if self.whiten:
            components = components * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance, 0.)
        precision = np.dot(components, components.T) / self.noise_variance
        precision.flat[::len(precision) + 1] += 1. / exp_var_diff
        precision = np.dot(components.T, np.dot(np.linalg.inv(precision),
                                                components))
        precision /= -(self.noise_variance ** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance
        return precision

    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        """
        if self.fitted is False:
            raise ValueError("The model should be fitted first.")
        X = X - self.mean
        return np.dot(X, self.components.T)

    def inverse_transform(self, X):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Note-  If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        if self.fitted is False:
            raise ValueError("The model should be fitted first.")
        if self.whiten:
            return np.dot(X, np.sqrt(self.explained_variances[:, np.newaxis]) *
                          self.components) + self.mean
        else:
            return np.dot(X, self.components) + self.mean
