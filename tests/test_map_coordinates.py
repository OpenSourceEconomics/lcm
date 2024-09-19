import numpy as np
import pytest
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates
from lcm.ndimage import map_coordinates as lcm_map_coordinates
from scipy.ndimage import map_coordinates as scipy_map_coordinates
from numpy.testing import assert_array_almost_equal
import jax.numpy as jnp


LIBRARIES = ["lcm", "jax", "scipy"]

def map_coordinates(library):
    mapping = {
        "lcm": lcm_map_coordinates,
        "jax": jax_map_coordinates,
        "scipy": scipy_map_coordinates
    }
    return mapping[library]

def asarray(library):
    mapping = {
        "lcm": jnp.asarray,
        "jax": jnp.asarray,
        "scipy": np.asarray
    }
    return mapping[library]


@pytest.mark.parametrize("library", LIBRARIES)
def test_map_coordinates01(library):
    data = np.array([[4, 1, 3, 2],
                       [7, 6, 8, 5],
                       [3, 5, 3, 6]])

    expected = np.array([[0, 0, 0, 0],
                           [0, 4, 1, 3],
                           [0, 7, 6, 8]])

    idx = np.indices(data.shape)
    idx -= 1
    idx = np.array(idx)
    
    data = asarray(library)(data)
    idx = asarray(library)(idx)

    out = map_coordinates(library)(data, idx, order=1)
    assert_array_almost_equal(out, expected)


# class TestMapCoordinates:

#     @pytest.mark.parametrize('order', range(0, 6))
#     @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
#     def test_map_coordinates01(self, order, dtype, xp):
#         if is_jax(xp) and order > 1:
#             pytest.xfail("jax map_coordinates requires order <= 1")

#         data = xp.asarray([[4, 1, 3, 2],
#                            [7, 6, 8, 5],
#                            [3, 5, 3, 6]])
#         expected = xp.asarray([[0, 0, 0, 0],
#                                [0, 4, 1, 3],
#                                [0, 7, 6, 8]])
#         isdtype = array_namespace(data).isdtype
#         if isdtype(data.dtype, 'complex floating'):
#             data = data - 1j * data
#             expected = expected - 1j * expected

#         idx = np.indices(data.shape)
#         idx -= 1
#         idx = xp.asarray(idx)

#         out = ndimage.map_coordinates(data, idx, order=order)
#         assert_array_almost_equal(out, expected)

#     @pytest.mark.parametrize('order', range(0, 6))
#     def test_map_coordinates02(self, order, xp):
#         if is_jax(xp):
#             if order > 1:
#                pytest.xfail("jax map_coordinates requires order <= 1")
#             if order == 1:
#                pytest.xfail("output differs. jax bug?")

#         data = xp.asarray([[4, 1, 3, 2],
#                            [7, 6, 8, 5],
#                            [3, 5, 3, 6]])
#         idx = np.indices(data.shape, np.float64)
#         idx -= 0.5
#         idx = xp.asarray(idx)

#         out1 = ndimage.shift(data, 0.5, order=order)
#         out2 = ndimage.map_coordinates(data, idx, order=order)
#         assert_array_almost_equal(out1, out2)

#     @skip_xp_backends("jax.numpy", reasons=["`order` is required in jax"],
#                       cpu_only=True, exceptions=['cupy', 'jax.numpy'],)
#     def test_map_coordinates03(self, xp):
#         data = _asarray([[4, 1, 3, 2],
#                          [7, 6, 8, 5],
#                          [3, 5, 3, 6]], order='F', xp=xp)
#         idx = np.indices(data.shape) - 1
#         idx = xp.asarray(idx)
#         out = ndimage.map_coordinates(data, idx)
#         expected = xp.asarray([[0, 0, 0, 0],
#                                [0, 4, 1, 3],
#                                [0, 7, 6, 8]])
#         assert_array_almost_equal(out, expected)
#         assert_array_almost_equal(out, ndimage.shift(data, (1, 1)))

#         idx = np.indices(data[::2, ...].shape) - 1
#         idx = xp.asarray(idx)
#         out = ndimage.map_coordinates(data[::2, ...], idx)
#         assert_array_almost_equal(out, xp.asarray([[0, 0, 0, 0],
#                                                    [0, 4, 1, 3]]))
#         assert_array_almost_equal(out, ndimage.shift(data[::2, ...], (1, 1)))

#         idx = np.indices(data[:, ::2].shape) - 1
#         idx = xp.asarray(idx)
#         out = ndimage.map_coordinates(data[:, ::2], idx)
#         assert_array_almost_equal(out, xp.asarray([[0, 0], [0, 4], [0, 7]]))
#         assert_array_almost_equal(out, ndimage.shift(data[:, ::2], (1, 1)))

#     @skip_xp_backends(np_only=True)
#     def test_map_coordinates_endianness_with_output_parameter(self, xp):
#         # output parameter given as array or dtype with either endianness
#         # see issue #4127
#         # NB: NumPy-only

#         data = np.asarray([[1, 2], [7, 6]])
#         expected = np.asarray([[0, 0], [0, 1]])
#         idx = np.indices(data.shape)
#         idx -= 1
#         for out in [
#             data.dtype,
#             data.dtype.newbyteorder(),
#             np.empty_like(expected),
#             np.empty_like(expected).astype(expected.dtype.newbyteorder())
#         ]:
#             returned = ndimage.map_coordinates(data, idx, output=out)
#             result = out if returned is None else returned
#             assert_array_almost_equal(result, expected)

#     @skip_xp_backends(np_only=True, reasons=['string `output` is numpy-specific'])
#     def test_map_coordinates_with_string_output(self, xp):
#         data = xp.asarray([[1]])
#         idx = np.indices(data.shape)
#         idx = xp.asarray(idx)
#         out = ndimage.map_coordinates(data, idx, output='f')
#         assert out.dtype is np.dtype('f')
#         assert_array_almost_equal(out, xp.asarray([[1]]))

#     @pytest.mark.skipif('win32' in sys.platform or np.intp(0).itemsize < 8,
#                         reason='do not run on 32 bit or windows '
#                                '(no sparse memory)')
#     def test_map_coordinates_large_data(self, xp):
#         # check crash on large data
#         try:
#             n = 30000
#             # a = xp.reshape(xp.empty(n**2, dtype=xp.float32), (n, n))
#             a = np.empty(n**2, dtype=np.float32).reshape(n, n)
#             # fill the part we might read
#             a[n - 3:, n - 3:] = 0
#             ndimage.map_coordinates(
#                 xp.asarray(a), xp.asarray([[n - 1.5], [n - 1.5]]), order=1
#             )
#         except MemoryError as e:
#             raise pytest.skip('Not enough memory available') from e