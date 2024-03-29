import numpy as np
from skimage.util.shape import view_as_windows
# ^ Uses np.lib.stride_tricks.as_strided so I don't have to.


def _fast_shift_batch(A, shifts, axis, fill_value=0):
    """
    Returns ndarray batch A where every image is shifted the corresponding amount according to shifts.
    Each image is shifted along axis "axis."
    The new space in each shifted ndarray is filled with fill_value.

    >>> a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    >>> shifts = (-1, 1)
    >>> _fast_shift_batch(a, shifts, axis=2, fill_value=np.nan)
    array([[[ 1.,  2.,  3., nan],
            [ 5.,  6.,  7., nan],
            [ 9., 10., 11., nan]],

           [[nan, 12., 13., 14.],
            [nan, 16., 17., 18.],
            [nan, 20., 21., 22.]]])
    """
    if A.ndim not in (3, 4):
        raise ValueError('A should be a batch of nxm or nxmxc images.')
    if axis not in (1, 2):
        raise ValueError('Batch shift axis should be either 1 or 2')
    shifts = np.array(shifts)
    fill = np.full(A.shape, fill_value=fill_value)
    A_ext = np.concatenate((fill, A, fill),  axis=axis)
    # The following returns correctly indexed and memory efficient window-views of A_ext.
    # np.squeeze is used to remove single-dim entries generated by
    # view_as_windows.
    return view_as_windows(A_ext, (1, *A.shape[1:])).squeeze()[
        np.arange(A.shape[0]), -shifts + A.shape[axis]
    ].reshape(A.shape)
    # Avoid doing any computation with view_as_windows as this will cause a lot of memory to be allocated.
    # See the 'notes' section in the docs for view_as_windows:
    # https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.view_as_windows


class NPImageDataGenerator(object):
    """
    A intermittent generator object which
    grabs data from tfds Dataset (as numpy) and
    does some preprocessing on X and Y
    before returning it in a generator fashion.
    """

    def __init__(self, dataset,
                 rescale=None, height_shift_range=None, width_shift_range=None,
                 shift_fill=0., final_shape=None,
                 num_classes=None):
        """
        rescale: scale factor of X values
        height_shift_range: number of pixels to shift images row-wise
        width_shift_range: number of pixels to shift images column-wise
        final_shape: final shape of X (Use if want to reshape X)
        num_classes: If None will not one hot encode, otherwise will based upon given value.
        """

        if height_shift_range is not None:
            if height_shift_range <= 0:
                raise ValueError(
                    'height_shift_range should be a positive pixel value')
        if width_shift_range is not None:
            if width_shift_range <= 0:
                raise ValueError(
                    'width_shift_range should be a positive pixel value')

        self._dataset = dataset
        self._rescale = rescale
        self._height_shift_range = height_shift_range
        self._width_shift_range = width_shift_range
        self._shift_fill = shift_fill
        self._final_shape = final_shape
        self._num_classes = num_classes

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _one_hot(self, Y):
        one_hot = np.zeros((Y.shape[0], self._num_classes))
        one_hot[np.arange(one_hot.shape[0]), Y] = 1
        return one_hot

    def _X_preprocessed(self, X):
        """
        Preprocesses X
        """
        if self._rescale is not None:
            X = X * self._rescale
        if self._height_shift_range is not None:
            shifts = np.random.randint(-self._height_shift_range,
                                       self._height_shift_range + 1, size=X.shape[0])
            X = _fast_shift_batch(
                X, shifts, axis=1, fill_value=self._shift_fill)
        if self._width_shift_range is not None:
            shifts = np.random.randint(-self._width_shift_range,
                                       self._width_shift_range + 1, size=X.shape[0])
            X = _fast_shift_batch(
                X, shifts, axis=2, fill_value=self._shift_fill)
        if self._final_shape is not None:
            X = X.reshape(final_shape)
        return X

    def _Y_preprocessed(self, Y):
        """
        Preprocesses Y
        """
        if self._num_classes is not None:
            Y = self._one_hot(Y)
        return Y

    def next(self):
        for (X, Y) in self._dataset:
            return self._X_preprocessed(X), self._Y_preprocessed(Y)
        raise StopIteration
