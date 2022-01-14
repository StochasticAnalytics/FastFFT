COMPILER := g++

CXX_FLAGS := -O3 -Wall -shared -std=c++17 -fPIC

SRCS := test_pybind11.cpp
# Needs to match the first arg to PYBIND11_MODULE(fastfft_test, m)
TARGETS := fastfft_test.so

# from $(python -m pybind11 --includes)
PYTHON_INCLUDES=-I/usr/include/python3.8 -I/groups/himesb/.myPyBind11/lib/python3.8/site-packages/pybind11/include

$(TARGETS): $(SRCS)
	$(COMPILER) $(CXX_FLAGS) $(PYTHON_INCLUDES) $^ -o $@

clean:
	rm -f *.so


