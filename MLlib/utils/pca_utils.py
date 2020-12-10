import numpy as np

class PCA_utils():
     
    def get_covariance(self):
        """Compute data covariance with the generative model.
     
        ``cov = components.T * S**2 * components + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        """
        
        if self.fitted == False:
            raise ValueError("The model should be fitted first.")
        
        components= self.components
        exp_var= self.explained_variances
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
        
        if self.fitted == False:
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
        precision = np.dot(components.T, np.dot(np.linalg.inv(precision), components))
        precision /= -(self.noise_variance** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance
        return precision
        
    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        X is projected on the first principal components previously extracted
        from a training set.
        """
        
        if self.fitted == False:
            raise ValueError("The model should be fitted first.")
        
        X = X - self.mean
        return np.dot(X, self.components.T)
    
    def inverse_transform(self, X):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        
        Note-  If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        
        if self.fitted == False:
            raise ValueError("The model should be fitted first.")
        
        if self.whiten:
            return np.dot(X, np.sqrt(self.explained_variances[:, np.newaxis]) * self.components) + self.mean
        else:
            return np.dot(X, self.components) + self.mean