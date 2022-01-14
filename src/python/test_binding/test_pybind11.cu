#include <pybind11/pybind11.h>

namespace py = pybind11;

template<class TypeOne, class TypeTwo>
__global__ void add_kernel(TypeOne one, TypeTwo two, TypeOne* retval) {

    // add then numbers and 1 to be sure it ran in this device code.
   one + TypeOne(two) + TypeOne(1.0f);
}

template<class TypeOne, class TypeTwo>
class TestClass {

    public:
        TestClass(TypeOne one, TypeTwo two) : one_(one), two_(two) {}

        TypeOne getOne() { return one_; }
        TypeTwo getTwo() { return two_; }

        TypeOne add(TypeOne i, TypeTwo j) {
            TypeOne* retval;
            cudaMallocManaged(&retval, sizeof(TypeOne));

            add_kernel<<<1, 1>>>(i, j, retval);
            return *retval;
        }

    private:
        TypeOne one_;
        TypeTwo two_;
};

template<typename typeOne, typename typeTwo>
void declare_array(py::module &m, const std::string &typestr) {
    using Class = TestClass<typeOne, typeTwo>;
    std::string pyclass_name = std::string("TestClass") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
    .def(py::init<typeOne, typeTwo>())
    .def("getOne", &TestClass<typeOne, typeTwo>::getOne) 
    .def("getTwo", &TestClass<typeOne, typeTwo>::getTwo) 
    .def("add", &TestClass<typeOne, typeTwo>::add);
}

PYBIND11_MODULE(fastfft_test, m) {
    
    declare_array<int, float>(m, "_int_float");
    declare_array<float, int>(m, "_float_int");
}
