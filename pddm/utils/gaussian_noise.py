import numpy as np
import numpy.random
import scipy.stats as ss

class GaussianNoiseGenerator:
    def __init__(self, norm_params, noise_dims):
        numpy.random.seed(0x5eed)
        self.norm_params = norm_params
        self.noise_dims = noise_dims

    def __call__(self):
        n_components = self.norm_params.shape[0]
        weights = np.ones(n_components, dtype=np.float64) / n_components
        mixture_idx = numpy.random.choice(len(weights), size=self.noise_dims, replace=True, p=weights)

        noise = numpy.fromiter((ss.norm.rvs(*(self.norm_params[i])) for i in mixture_idx),
                       dtype=np.float64)

        return noise

if __name__ == "__main__":
    # norm_params = np.array([[-5, 1],
    #                         [6, 1.3]])

    norm_params = np.array([[5, 1.5]])

    g = GaussianNoiseGenerator(norm_params, 7)
    for i in range(10):
        print(g())

