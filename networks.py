import numpy as np
import warnings
import matplotlib.pyplot as plt
from functools import reduce
from typing import Literal, Optional
import sys

class MAP ():

    def __init__(self, shape: tuple, dim: int, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random'):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method to use for initializing the weights of the SOM

        Returns
        -------
        None

        """
        self.shape = shape
        self.dim = dim
        self.total_iterations = 0

        if len(shape) > 3:
            warnings.warn(
                'Warning: shape tuple is greater than length 3. The map is not in a visualizable space.')

        if (len(shape) > dim):
            raise Exception(
                'Error: shape tuple vector space is greater than input dimension.')

        # Fixed initialization
        if init_method == 'fixed':
            self.weights = np.ones(shape + (dim,))*0.5

        # Regular grid initialization
        elif init_method == 'regular':

            self.weights = np.zeros(shape + (dim,))
            for i in range(shape[0]):
                self.weights[i, :, 0] = np.linspace(0, 1, shape[1])
                self.weights[:, i, 1] = np.linspace(0, 1, shape[1])

        # Random initialization
        else:
            self.weights = np.random.random(shape + (dim,))

    def learn(self, data: np.ndarray, epochs: int = 100, plot: bool = True, shuffle: bool = True, random_state=None) -> None:
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.

        Parameters
        ----------
        data : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        epochs : int, default=100
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.
        plot : bool, default True
            Whether or not to plot the weights of the SOM while training.
        random_state : int, default None
            The seed to use for the random number generator. If None, the
            random number generator is the RandomState instance used by
            np.random.

        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """

        self.total_iterations = epochs * data.shape[0]
        n_samples = data.shape[0]
        global_iter_count = 0

        # create a 2D or 3D scatter plot
        fig = plt.figure()

        fig.canvas.mpl_connect('close_event', self.__del__)

        if plot:
            if self.dim == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter([], [], [])
            else:
                ax = fig.add_subplot(111)
                ax.scatter([], [])

        for epoch in range(epochs):
            # Shuffle indices
            if shuffle:
                rng = np.random.default_rng(random_state)
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            for idx in indices:
                input = data[idx]
                # Do one step of training
                self.step(input)
                plot_weights = self.weights.reshape(
                    reduce(lambda x, y: x*y, self.shape), self.dim)
                global_iter_count += 1
                if (plot):
                    if self.dim == 1:
                        ax.scatter(plot_weights[:, 0], c='b')
                    elif self.dim == 2:
                        ax.scatter(plot_weights[:, 0],
                                   plot_weights[:, 1], c='b')
                    elif self.dim == 3:
                        ax.scatter(
                            plot_weights[:, 0], plot_weights[:, 1], plot_weights[:, 2], c='b')
                    else:
                        Exception(
                            'Error: input dimension in not in a visualizable space')
                    # set the plot title
                    ax.set_title("Epoch: "+str(epoch+1) +
                                 ", Iteration: "+str(global_iter_count))
                    ax.collections[0].remove()
                    plt.draw()
                    plt.pause(0.001)
            QE = self.get_quantization_error(data)

            # write the quantization error to a file
            with open(f'QE_{self.shape}_{self.dim}.txt', 'a') as f:
                f.write(str(QE) + '\n')

    def get_quantization_error(self, data):
        n_samples = data.shape[0]
        indices = np.arange(n_samples)

        distances = []
        for idx in indices:
            input = data[idx]

            # calculate euclidean distance between input and weights
            dist = np.linalg.norm(self.weights - input, axis=-1)
            bmu_distance = np.min(dist)
            distances.append(bmu_distance)

        return np.average(np.array(distances))

    def __del__(self, event=None):
        print('Deleting SOM object ' + str(event))
        plt.close()
        sys.exit()


"""
Self Organising Maps - Teuvo Kohonen
"""
class SOM(MAP):

    def __init__(self, shape: tuple, dim: int, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', sigma_i: float = 10.00, sigma_f: float = 0.010, lrate_i: float = 0.500, lrate_f: float = 0.005):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method to use for initializing the weights of the SOM
        sigma_i : float, default=10.00
            The initial value of the sigma parameter for the Gaussian
            neighborhood function.
        sigma_f : float, default=0.010
            The final value of the sigma parameter for the Gaussian
            neighborhood function.
        lrate_i : float, default=0.500
            The initial value of the learning rate.
        lrate_f : float, default=0.005
            The final value of the learning rate.

        Returns
        -------
        None

        """
        super().__init__(shape, dim, init_method)
        self.sigma_i = sigma_i
        self.sigma_f = sigma_f
        self.lrate_i = lrate_i
        self.lrate_f = lrate_f
        self.global_t = 0

    def step(self, x: np.ndarray) -> None:
        """
        Take a single step of training on the input vector x.

        Parameters
        ----------
        x : ndarray
            The input vector. Must have shape (self.dim,).

        Returns
        -------
        None

        """

        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        # update learning rate and sigma values
        t = self.global_t/self.total_iterations
        lrate = self.lrate_i*(self.lrate_f/self.lrate_i)**t
        sigma = self.sigma_i*(self.sigma_f/self.sigma_i)**t

        # update weights
        eta = np.repeat((lrate * np.exp(-1*dist_to_bmu**2 /
                        (2 * sigma**2)))[..., np.newaxis], self.dim, axis=-1)
        self.weights += eta * (x - self.weights)

        self.global_t += 1

"""
DSOM - N. Rougier
"""
class DSOM(MAP):

    def __init__(self, shape: tuple, dim: int, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', sigma: float = 1.00, lrate: float = 0.500):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method used for initializing the weights of the SOM
        sigma : float, default=1.00
            The initial value of the sigma parameter for the Gaussian
            neighborhood function.
        lrate : float, default=0.500
            The initial value of the learning rate.

        Returns
        -------
        None

        """

        super().__init__(shape, dim, init_method)
        self.sigma = sigma
        self.lrate = lrate
        self.max = 0

    def step(self, x: np.ndarray) -> None:
        """
        Take a single step of training on the input vector x.

        Parameters
        ----------
        x : ndarray
            The input vector. Must have shape (self.dim,).

        Returns
        -------
        None
        """
        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        self.max = max(self.max, np.max(dist))
        normalised_norm = dist/self.max

        # update learning rate and sigma values
        sigma = self.sigma**2 * normalised_norm**2
        lrate = self.lrate * normalised_norm

        # update the weights
        eta = np.repeat((lrate * np.exp(-1*dist_to_bmu**2 /
                        (2 * sigma**2)))[..., np.newaxis], self.dim, axis=-1)
        self.weights += eta * (x - self.weights)

"""
A New Self-Organizing Map with Continuous Learning Capability - H. Hikawa
"""
class HikawaSOM(MAP):

    def __init__(self, shape: tuple, dim: int, k: float, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', sigma: float = 1.00, lrate: float = 0.500):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        k : float
            The weight for new bmu distance in exponential average
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method to use for initializing the weights of the SOM
        sigma : float, default=1.00
            The initial value of the sigma parameter for the Gaussian
            neighborhood function.
        lrate : float, default=0.500
            The initial value of the learning rate.

        Returns
        -------
        None

        """

        super().__init__(shape, dim, init_method)
        self.sigma = sigma
        self.lrate = lrate
        self.exp_avg = np.NaN
        self.k = k

    def step(self, x: np.ndarray) -> None:
        """
        Take a single step of training on the input vector x.

        Parameters
        ----------
        x : ndarray
            The input vector. Must have shape (self.dim,).

        Returns
        -------
        None
        """

        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        bmu_distance = dist[bmu_location]

        if (np.isnan(self.exp_avg)):
            self.exp_avg = bmu_distance

        # distance from the bmu to all other nodes in the grid
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        # relative distance to the bmu
        relative_dist = bmu_distance / self.exp_avg

        sigma = self.sigma * relative_dist
        lrate = self.lrate * relative_dist

        # update the weights
        eta = np.repeat((lrate * np.exp(-1*dist_to_bmu**2 /
                        (2 * sigma**2)))[..., np.newaxis], self.dim, axis=-1)
        self.weights += eta * (x - self.weights)

        # update exponential average
        self.exp_avg = self.k * bmu_distance + (1 - self.k) * self.exp_avg

"""
Brain-inspired self-organizing model for incremental learning - K. Gunawardena
"""
class BI_1SOM(MAP):

    def __init__(self, shape: tuple, dim: int, threshold: float, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', sigma: float = 1.00, lrate: float = 0.500):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        threshold : float
            The threshold for the cosine similarity between the adaptation of a node and the change in the weights of the node
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method to use for initializing the weights of the SOM
        sigma : float, default=1.00
            The initial value of the sigma parameter for the Gaussian
            neighborhood function.
        lrate : float, default=0.500
            The initial value of the learning rate.

        Returns
        -------
        None

        """
        super().__init__(shape, dim, init_method)

        if (threshold < 0 or threshold > 1):
            raise ValueError("Threshold must be between 0 and 1")

        self.threshold = threshold
        self.sigma = sigma
        self.lrate = lrate
        self.adaptations = np.full(self.shape + (self.dim,), np.nan)

    def is_excited(self, index: tuple, adaptation: np.ndarray) -> bool:
        """
        Check if the index is excited.
        Parameters
        ----------
        index : tuple
            The index to check.
        adaptation : np.ndarray
            The adaptation vector to check against.
        Returns
        -------
        bool
            True if the index is excited, False otherwise.
        """
        # check cosine similarity between adaptation and adaptation at index
        return np.dot(adaptation, self.adaptations[index]) > self.threshold

    def get_neighbors(self, location: tuple) -> set:
        """
        Get the neighbors of the index inside the map.
        Parameters
        ----------
        index : tuple
            The index to get the neighbors of.
        Returns
        -------
        neighbors : set
            The set of neighbors.

        """
        neighbors = set()
        for index in np.ndindex(*self.shape):
            if np.abs(np.array(index) - np.array(location)).sum() == 1:
                neighbors.add(index)

        return neighbors

    def step(self, x: np.ndarray) -> None:
        """
        Take a single step of training on the input vector x.

        Parameters
        ----------
        x : ndarray
            The input vector. Must have shape (self.dim,).

        Returns
        -------
        None
        """
        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        neighbors = set()
        visited = set()

        # add the bmu to the neighbors
        neighbors.add(bmu_location)

        # while there are still neighbors to visit
        while (len(neighbors) > 0):
            # get the next neighbor
            current = neighbors.pop()

            # if the neighbor has not been visited
            if (current not in visited):
                # mark the neighbor as visited
                visited.add(current)

                # calculate the change in weights
                eta = np.repeat((self.lrate * np.exp(-dist_to_bmu **
                                2/self.sigma))[..., np.newaxis], self.dim, axis=-1)
                delta = eta * (x - self.weights)

                norm_of_current_delta = np.linalg.norm(delta[current])
                norm_of_current_adaptation = np.linalg.norm(
                    self.adaptations[current])

                if (current == bmu_location or self.is_excited(bmu_location, current, x) or norm_of_current_delta > norm_of_current_adaptation):
                    # update the weights
                    self.weights += delta
                    self.adaptations[current] += delta[current]

                    # add the neighbors of the current node to the neighbors
                    neighbors.update(self.get_neighbors(current))

                else:
                    self.adaptations[current] += delta[current]

"""
Improving Quantization Quality in Brain-Inspired Self-organization for Non-stationary Data Spaces - K. Gunawardena
"""
class BI_2SOM(MAP):

    def __init__(self, shape: tuple, dim: int, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', sigma: float = 1.00, lrate: float = 0.500):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the SOM. For example, a 2D SOM with 10x10 nodes would
            have shape (10, 10).
        dim : int
            The dimensionality of the input space.
        init_method : str, default='random' (other options: 'fixed', 'regular')
            The method to use for initializing the weights of the SOM
        sigma : float, default=1.00
            The initial value of the sigma parameter for the Gaussian
            neighborhood function.
        lrate : float, default=0.500
            The initial value of the learning rate.

        Returns
        -------
        None

        """
        super().__init__(shape, dim, init_method)
        self.sigma = sigma
        self.lrate = lrate
        self.receptives = np.full(self.shape, np.nan)
        self.n = np.ones(self.shape)

    def update_receptive_distances(self, bmu_location: tuple, index: tuple, input: np.ndarray, is_excited: bool) -> None:
        """
        Update the receptive distance of the index in the map.
        Parameters
        ----------
        bmu_location : tuple
            The location of the best matching unit.
        index : tuple
            The index to update the receptive distance of.
        input : np.ndarray
            The input vector.
        is_excited : bool
            True if the index is excited, False otherwise.

        Returns
        -------
        None
        """
        dist_to_bmu = np.linalg.norm(index - bmu_location)
        dist_to_input = np.linalg.norm(input - self.weights[index])

        if (np.isnan(self.receptives[index])):
            self.receptives[index] = (
                2 - np.exp(-1*dist_to_bmu**2/self.sigma)) * dist_to_input
        else:
            if (is_excited):
                self.receptives[index] = (
                    self.receptives[index] + (2 - np.exp(-1*dist_to_bmu**2/self.sigma)) * dist_to_input) / 2
            else:
                self.receptives[index] = (self.receptives[index] * self.n[index] + (
                    2 - np.exp(-1*dist_to_bmu**2/self.sigma)) * dist_to_input) / self.n[index]

    def is_excited(self, bmu_location: tuple, index: tuple, input: np.ndarray) -> bool:
        """
        Check if the index is excited.
        Parameters
        ----------
        bmu_location : tuple
            The location of the best matching unit.
        index : tuple
            The index to check if it is excited.
        input : np.ndarray
            The input vector.

        Returns
        -------
        bool
            True if the index is excited, False otherwise.
        """
        dist_to_bmu = np.linalg.norm(index - bmu_location)
        dist_to_input = np.linalg.norm(input - self.weights[index])

        if (np.isnan(self.receptives[index])):
            return True
        else:
            return self.receptives[index] > (2 - np.exp(-1*dist_to_bmu**2/self.sigma)) * dist_to_input

    def get_neighbors(self, location: tuple) -> set:
        """
        Get the direct neighbors of the index inside the map.
        Parameters
        ----------
        index : tuple
            The index to get the neighbors of.
        Returns
        -------
        neighbors : set
            The set of neighbors.

        """
        neighbors = set()
        for index in np.ndindex(*self.shape):
            if np.abs(np.array(index) - np.array(location)).sum() == 1:
                neighbors.add(index)

        return neighbors

    def step(self, x: np.ndarray) -> None:
        """
        Take a single step of training on the input vector x.
        Parameters
        ----------
        x : np.ndarray
            The input vector.

        Returns
        -------
        None
        """
        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        neighbors = set()
        visited = set()

        # add the bmu to the neighbors
        neighbors.add(bmu_location)

        # while there are still neighbors to visit
        while (len(neighbors) > 0):
            # get the next neighbor
            current = neighbors.pop()

            # if the neighbor has not been visited
            if (current not in visited):
                # mark the neighbor as visited
                visited.add(current)
                self.n[current] += 1

                if (current == bmu_location or self.is_excited(bmu_location, current, x)):
                    # update the weights
                    eta = np.repeat(
                        (self.lrate * np.exp(-1*dist_to_bmu**2/self.sigma))[..., np.newaxis], self.dim, axis=-1)
                    self.weights += eta * (x - self.weights)

                    # update the receptive distances
                    self.update_receptive_distances(
                        bmu_location, current, x, True)

                    # add the neighbors of the current node to the neighbors
                    neighbors.update(self.get_neighbors(current))

                else:
                    # update the receptive distances
                    self.update_receptive_distances(
                        bmu_location, current, x, False)

"""
PLSOM - E. Berglund
"""
class PLSOM(MAP):

    def __init__(self, shape: tuple, dim: int, beeta: float = 0.2, theta_min: float = 1, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', lrate: float = 0.500):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the map.
        dim : int
            The dimension of the input vectors.
        beeta : float
            The beeta parameter.
        theta_min : float
            The theta_min parameter.
        init_method : str
            The initialization method.
        lrate : float
            The learning rate.
        """

        super().__init__(shape, dim, init_method)
        self.theta_min = theta_min
        self.beeta = beeta
        self.lrate = lrate
        self.scaling_variable = 0
        self.r = 0

    def step(self, x: np.ndarray):
        """
        Take a single step of training on the input vector x.
        Parameters
        ----------
        x : np.ndarray
            The input vector.

        Returns
        -------
        None
        """
        # calculate euclidean distance between input and weights
        dist = np.linalg.norm(self.weights - x, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        bmu_distance = dist[bmu_location]

        self.r = max(self.r, bmu_distance)
        self.scaling_variable = bmu_distance / self.r

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        # update learning rate and sigma values
        lrate = self.scaling_variable
        sigma = (self.beeta - self.theta_min) * \
            self.scaling_variable + self.theta_min

        # update weights
        eta = np.repeat((lrate * np.exp(-1*dist_to_bmu**2 /
                        (sigma**2)))[..., np.newaxis], self.dim, axis=-1)
        self.weights += eta * (x - self.weights)

"""
TASOM - H. Shah-Hosseini
"""
class TASOM(MAP):

    def __init__(self, shape: tuple, dim: int, init_method: Optional[Literal['fixed', 'random', 'regular']] = 'random', s_f: float = 0.5, s_g: float = 0.5, lrate: float = 0.9, alpha: float = 0.5, beta: float = 0.5, alpha_s: float = 0.5, beta_s: float = 0.5):
        """
        Parameters
        ----------
        shape : tuple
            The shape of the map.
        dim : int
            The dimension of the input vectors.
        init_method : str
            The initialization method.
        s_f : float
            The s_f parameter. default: 0.5
        s_g : float
            The s_g parameter. default: 0.5
        lrate : float
            The learning rate. default: 0.9 (close to unity)
        alpha : float
            The alpha parameter. default: 0.5
        beta : float
            The beta parameter. default: 0.5
        alpha_s : float
            The alpha_s parameter. default: 0.5
        beta_s : float
            The beta_s parameter. default: 0.5

        Returns
        -------
        None
        """
        if (len(shape)) > 2:
            raise ValueError(
                'The shape of the map for TASOM must be 2D or 1D.')

        super().__init__(shape, dim, init_method)
        self.alpha = alpha
        self.beta = beta
        self.alpha_s = alpha_s
        self.beta_s = beta_s
        self.s_f = s_f
        self.s_g = s_g
        self.lrate = np.full(shape, lrate)
        self.s = np.random.uniform(0, 1, dim)
        self.R = np.full(shape, 0)
        self.E = np.random.rand(dim)
        self.E2 = np.random.rand(dim)

    def get_neighbors(self, location: tuple) -> set:
        """
        Get the neighbors of the index inside the map.
        Parameters
        ----------
        index : tuple
            The index to get the neighbors of.
        Returns
        -------
        neighbors : set
            The set of neighbors.

        """
        neighbors = set()
        for index in np.ndindex(*self.shape):
            if np.abs(np.array(index) - np.array(location)).sum() == 1:
                neighbors.add(index)

        return neighbors

    def g(self, x: float) -> float:
        """
        scalar function for which dg(z)/dz >= 0 for z > 0,
        and is used for normalization of the weight distances.
        for 1D maps of N, g(0) = 0 and 0 <= g(z) <= N (number of nodes)
        for 2D map of N x N, 0 <= g(z) <= sqrt(2)*N
        Parameters
        ----------
        x : float
            The s parameter.
        Returns
        -------
        float
            The value of the g function.
        """
        network_dim = len(self.shape)
        if (network_dim == 1):
            return (reduce(lambda x, y: x*y, self.shape) - 1)*(x/(x+1))
        elif (network_dim == 2):
            return (np.sqrt(reduce(lambda x, y: x*y, self.shape)) * np.sqrt(2.0) - 1)*(x/(x+1))

    def f(self, x: float) -> float:
        """
        scalar function for which df(z)/dz >= 0 for z > 0,
        0 <= f(z) <= 1 and f(0) = 0 for z > 0
        Parameters
        ----------
        x : float
            The s parameter.
        Returns
        -------
        float
            The value of the f function.
        """
        return x/(x+1)

    def step(self, x: np.ndarray):
        """
        Take a single step of training on the input vector x.
        Parameters
        ----------
        x : np.ndarray
            The input vector.

        Returns
        -------
        None
        """
        # calculate euclidean distance between input and weights
        dist = np.linalg.norm((self.weights - x)/self.s, axis=-1)

        # find the best matching unit
        bmu_location = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # get direct neighbors of the best matching unit
        direct_neighbors = self.get_neighbors(bmu_location)
        number_of_neighbors = len(direct_neighbors)

        # sum of the distances from the best matching unit to its neighbors
        distance_from_bmu_to_neighbors = reduce(lambda x, y: x+y, [np.linalg.norm((
            self.weights[bmu_location] - self.weights[neighbor])/self.s) for neighbor in direct_neighbors])

        # update the R value of the best matching unit
        self.R[bmu_location] += self.beta * \
            (self.g(self.s_g * number_of_neighbors**-1 *
             distance_from_bmu_to_neighbors) - self.R[bmu_location])

        # distance from the best matching unit to all other nodes in the map
        dist_to_bmu = np.linalg.norm(
            np.stack(np.indices(self.shape), axis=-1) - bmu_location, axis=-1)

        # find the nodes dist_to_bmu less than R value of the best matching unit
        nodes_to_update = np.where(dist_to_bmu < self.R[bmu_location])

        # update the learning rate of the nodes dist_to_bmu less than R value of the best matching unit
        self.lrate[nodes_to_update] += self.alpha * \
            (self.f(dist[nodes_to_update] / self.s_f) -
             self.lrate[nodes_to_update])

        # update the weights of the node
        eta = np.repeat((self.lrate[nodes_to_update])
                        [..., np.newaxis], self.dim, axis=-1)
        self.weights[nodes_to_update] += eta * \
            (x - self.weights[nodes_to_update])

        self.E2 += self.alpha_s * (x**2 - self.E2)
        self.E += self.beta_s * (x - self.E)

        self.s = np.sqrt(np.maximum(0, self.E2 - self.E**2))
