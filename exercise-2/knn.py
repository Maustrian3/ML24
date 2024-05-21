import numpy as np
from scipy.spatial.distance import euclidean, minkowski, cosine, cityblock

# sources: https://en.wikipedia.org/wiki/K-d_tree
class CustomKNeighborsRegressor:
    def __init__(self, n_neighbors=5, algorithm='brute', metric='minkowski', p=2, leaf_size=1):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm  #brute or kd_tree
        self.leaf_size = leaf_size

        if metric == 'euclidean':
            self.metric = euclidean
        elif metric == 'minkowski':
            def my_minkowski(x, y):
                return minkowski(x, y, p)
            self.metric = my_minkowski
        elif metric == 'cosine':
            self.metric = cosine
        elif metric == 'cityblock':
            self.metric = cityblock

        self.X_train = None
        self.y_train = None
        self.kd_tree = None


    def fit(self, X, y):
        if self.algorithm == 'brute':
            self.X_train = X
            self.y_train = y
        elif self.algorithm == 'kd_tree':
            self.__fit_kd_tree(X, y)

    def __fit_kd_tree(self, X, y):
        points = [self._Point(x, y[i]) for i, x in enumerate(X)]
        self.kd_tree = self._build_kd_tree(points, leaf_size=self.leaf_size)

    def predict(self, X):
        if self.algorithm == 'brute':
            return self.__predict_brute(X)
        elif self.algorithm == 'kd_tree':
            return self.__predict_kd_tree(X)

    def __predict_brute(self, X):
        y_pred = []

        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self.metric(x, x_train)
                target = self.y_train[i]
                distances.append((dist, target))

            # sort the distances by the first element (distance)
            distances.sort(key=lambda key: key[0])
            # take the k nearest neighbors
            k_nearest = distances[:self.n_neighbors]

            # find mean of the k nearest neighbors
            mean = sum([x[1] for x in k_nearest]) / self.n_neighbors

            y_pred.append(mean)

        # return ndarray (for compatibility with sklearn)
        return np.array(y_pred)

    def __predict_kd_tree(self, X):
        y_pred = []

        for x in X:
            target = self._search_kd_tree(x)
            y_pred.append(np.mean(target))

        return np.array(y_pred)


    class _Point:
        def __init__(self, coords, target):
            self.coords = coords
            self.target = target


    class _TreeNode:
        def __init__(self, point, left=None, right=None, axis=0):
            self.value = point
            self.left = left
            self.right = right
            self.axis = axis


    def _build_kd_tree(self, points, leaf_size, depth=0):
        if len(points) == 0:
            return None

        dimension = len(points[0].coords)
        # change axis in each level of the tree
        axis = depth % dimension

        if leaf_size != 1 and len(points) <= leaf_size:
            return self._TreeNode(point=points,
                             left=None,
                             right=None,
                             axis=axis)

        # sort the points and choose the median as pivot element
        points.sort(key=lambda point: point.coords[axis])
        median = len(points) // 2

        return self._TreeNode(
            point=points[median],
            left=self._build_kd_tree(points[:median], depth + 1),
            right=self._build_kd_tree(points[median + 1:], depth + 1),
            axis=axis
        )

    def _search_kd_tree(self, target):
        if self.kd_tree is None:
            return None

        best = _BestQueue(self.n_neighbors)
        self.search(self.kd_tree, target, best)

        return [point[0].target for point in best.get()]

    def search(self, node, target, best):
        if node is None:
            return

        if isinstance(node.value, list):
            # do brute search in leaf
            for point in node.value:
                dist = self.metric(target, point.coords)
                best.add(point, dist)
            return

        # from here on: not a leaf
        point = node.value
        dist = self.metric(target, point.coords)
        best.add(point, dist)

        next_branch = None
        opposite_branch = None

        if target[node.axis] < point.coords[node.axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left

        self.search(next_branch, target, best)

        # if not enough points found already
        # or the distance to the opposite branch is smaller than the farthest point found so far
        farthest_so_far = best.get()[-1][1]
        if best.get_length() < self.n_neighbors or abs(point.coords[node.axis] - target[node.axis]) < farthest_so_far:
            self.search(opposite_branch, target, best)



class _BestQueue:
    """
    A queue that keeps the k best elements.
    """
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, point, dist):
        """
        Add a point to the queue.
        Only the `size` best points are kept.
        :param point: point to add
        :param dist: distance of the point that is used for sorting
        """
        if len(self.data) < self.size:
            self.data.append((point, dist))
            self.data.sort(key=lambda x: x[1])
            return
        elif dist < self.data[-1][1]:
            self.data.append((point, dist))
            self.data.sort(key=lambda x: x[1])
            self.data = self.data[:self.size]

    def get(self):
        return self.data

    def get_length(self):
        return len(self.data)
