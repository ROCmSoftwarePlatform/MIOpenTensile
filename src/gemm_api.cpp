#include <miopentensile/gemm.h>
#include <Tensile/Tensile.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

template<class T>
auto& deref(T* x)
{
    if (x == nullptr)
        throw std::runtime_error("Dereference null pointer");
    return *x;
}

const auto& library()
{
    static auto result = Tensile::EmbeddedLibrary<Tensile::ContractionProblem>::NewLibrary("miopen_tensile_kernels");
    return *result;
}

bool is_transposed(const miopen_tensile_matrix& a)
{
    return a.strides[1] > a.strides[0];
}

size_t get_ld(const miopen_tensile_matrix& a)
{
    return a.strides[is_transposed(a) ? 1 : 0];
}

Tensile::ContractionProblem create_tensile_problem(miopen_tensile_matrix& a, miopen_tensile_matrix& b, miopen_tensile_matrix& c)
{
    return Tensile::ContractionProblem::GEMM(is_transposed(a), is_transposed(b), a.lens[0], b.lens[1], a.lens[1], get_ld(a), get_ld(b), get_ld(c), 0.0, false, 1);
}

extern "C" {

miopen_tensile_status miopen_tensile_gemm(hipStream_t stream, miopen_tensile_matrix* a, miopen_tensile_matrix* b, miopen_tensile_matrix* c)
{
    auto problem = create_tensile_problem(deref(a), deref(b), deref(c));
    auto hardware = Tensile::hip::GetCurrentDevice();
    auto solution = library().findBestSolution(problem, *hardware);
    if (not solution)
    {
        std::cerr << "No solution found." << std::endl;
        return miopen_tensile_status_no_solution;
    }
    Tensile::TypedContractionInputs<float> inputs;
    inputs.a = reinterpret_cast<const float*>(a->data);
    inputs.b = reinterpret_cast<const float*>(b->data);
    inputs.c = reinterpret_cast<const float*>(c->data);
    inputs.d = reinterpret_cast<float*>(c->data);
    auto kernels = solution->solve(problem, inputs, *hardware);
    return miopen_tensile_status_success;
}

}