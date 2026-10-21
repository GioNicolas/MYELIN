import builtins
import functools
import itertools
import math
import numbers
import operator
import sys
import warnings

import numpy as np
from numpy.exceptions import AxisError

from . import multiarray, numerictypes, numerictypes as nt, overrides, shape_base, umath
from ._ufunc_config import errstate
from .multiarray import (  # noqa: F401
    ALLOW_THREADS,
    BUFSIZE,
    CLIP,
    MAXDIMS,
    MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT,
    RAISE,
    WRAP,
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    broadcast,
    can_cast,
    concatenate,
    copyto,
    dot,
    dtype,
    empty,
    empty_like,
    flatiter,
    from_dlpack,
    frombuffer,
    fromfile,
    fromiter,
    fromstring,
    inner,
    lexsort,
    matmul,
    may_share_memory,
    min_scalar_type,
    ndarray,
    nditer,
    nested_iters,
    normalize_axis_index,
    promote_types,
    putmask,
    result_type,
    shares_memory,
    vdot,
    vecdot,
    where,
    zeros,
)
from .overrides import finalize_array_function_like, set_module
from .umath import NAN, PINF, invert, multiply, sin

bitwise_not = invert
ufunc = type(sin)
newaxis = None

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'newaxis', 'ndarray', 'flatiter', 'nditer', 'nested_iters', 'ufunc',
    'arange', 'array', 'asarray', 'asanyarray', 'ascontiguousarray',
    'asfortranarray', 'zeros', 'count_nonzero', 'empty', 'broadcast', 'dtype',
    'fromstring', 'fromfile', 'frombuffer', 'from_dlpack', 'where',
    'argwhere', 'copyto', 'concatenate', 'lexsort', 'astype',
    'can_cast', 'promote_types', 'min_scalar_type',
    'result_type', 'isfortran', 'empty_like', 'zeros_like', 'ones_like',
    'correlate', 'convolve', 'inner', 'dot', 'outer', 'vdot', 'roll',
    'rollaxis', 'moveaxis', 'cross', 'tensordot', 'little_endian',
    'fromiter', 'array_equal', 'array_equiv', 'indices', 'fromfunction',
    'isclose', 'isscalar', 'binary_repr', 'base_repr', 'ones',
    'identity', 'allclose', 'putmask',
    'flatnonzero', 'inf', 'nan', 'False_', 'True_', 'bitwise_not',
    'full', 'full_like', 'matmul', 'vecdot', 'shares_memory',
    'may_share_memory']


def _zeros_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_zeros_like_dispatcher)
def zeros_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=np.float64)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # needed instead of a 0 to get same result as zeros for string dtypes
    z = zeros(1, dtype=res.dtype)
    multiarray.copyto(res, z, casting='unsafe')
    return res


@finalize_array_function_like
@set_module('numpy')
def ones(shape, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=np.int_)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[1.],
           [1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])

    """
    if like is not None:
        return _ones_with_like(
            like, shape, dtype=dtype, order=order, device=device
        )

    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, 1, casting='unsafe')
    return a


_ones_with_like = array_function_dispatch()(ones)


def _ones_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_ones_like_dispatcher)
def ones_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = np.arange(3, dtype=np.float64)
    >>> y
    array([0., 1., 2.])
    >>> np.ones_like(y)
    array([1.,  1.,  1.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, 1, casting='unsafe')
    return res


def _full_dispatcher(
    shape, fill_value, dtype=None, order=None, *, device=None, like=None
):
    return (like,)


@finalize_array_function_like
@set_module('numpy')
def full(shape, fill_value, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         ``np.array(fill_value).dtype``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> np.full((2, 2), np.inf)
    array([[inf, inf],
           [inf, inf]])
    >>> np.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])

    >>> np.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])

    """
    if like is not None:
        return _full_with_like(
            like, shape, fill_value, dtype=dtype, order=order, device=device
        )

    if dtype is None:
        fill_value = asarray(fill_value)
        dtype = fill_value.dtype
    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, fill_value, casting='unsafe')
    return a


_full_with_like = array_function_dispatch()(full)


def _full_like_dispatcher(
    a, fill_value, dtype=None, order=None, subok=None, shape=None,
    *, device=None
):
    return (a,)


@array_function_dispatch(_full_like_dispatcher)
def full_like(
    a, fill_value, dtype=None, order='K', subok=True, shape=None,
    *, device=None
):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : array_like
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6, dtype=np.int_)
    >>> np.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])
    >>> np.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0])
    >>> np.full_like(x, 0.1, dtype=np.float64)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    >>> np.full_like(x, np.nan, dtype=np.float64)
    array([nan, nan, nan, nan, nan, nan])

    >>> y = np.arange(6, dtype=np.float64)
    >>> np.full_like(y, 0.1)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    >>> y = np.zeros([2, 2, 3], dtype=np.int_)
    >>> np.full_like(y, [0, 0, 255])
    array([[[  0,   0, 255],
            [  0,   0, 255]],
           [[  0,   0, 255],
            [  0,   0, 255]]])
    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, fill_value, casting='unsafe')
    return res


def _count_nonzero_dispatcher(a, axis=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_count_nonzero_dispatcher)
def count_nonzero(a, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array ``a``.

    A non-zero value is one that evaluates to truthful in a boolean
    context, including any non-zero number and any string that
    is not empty. This function recursively counts how many elements
    in ``a`` (and its sub-arrays) are non-zero values.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.
    keepdims : bool, optional
        If this is set to True, the axes that are counted are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    count : int or array of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    nonzero : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> import numpy as np
    >>> np.count_nonzero(np.eye(4))
    np.int64(4)
    >>> a = np.array([[0, 1, 7, 0],
    ...               [3, 0, 2, 19]])
    >>> np.count_nonzero(a)
    np.int64(5)
    >>> np.count_nonzero(a, axis=0)
    array([1, 1, 2, 1])
    >>> np.count_nonzero(a, axis=1)
    array([2, 3])
    >>> np.count_nonzero(a, axis=1, keepdims=True)
    array([[2],
           [3]])
    """
    if axis is None and not keepdims:
        return multiarray.count_nonzero(a)

    a = asanyarray(a)

    # This is a performance optimization for character dtypes
