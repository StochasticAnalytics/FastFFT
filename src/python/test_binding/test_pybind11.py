import fastfft_test as f
import cupy as cp

print('Entering test pybin11')
t = f.TestClass_float_float(3.4,3)

print('Member variable one has value {}'.format(t.getOne()))

print('Output of the add function is {}'.format(t.add(1.25,3)))

a = cp.array([1,2,3])

print('Array, before'.format(a))

print('Output of the cupy reduce functions {}'.format(t.sum_cupy_array(a,3)))
