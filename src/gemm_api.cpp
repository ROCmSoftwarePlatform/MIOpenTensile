#include <miopentensile/gemm.h>
#include <Tensile/Tensile.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <dlfcn.h>
#include <glob.h>

std::vector<std::string> glob_files(const std::string& s)
{
    std::vector<std::string> result;
    glob_t raw_glob_result;
    int e = glob(s.c_str(), GLOB_TILDE_CHECK | GLOB_NOSORT, nullptr, &raw_glob_result);
    std::shared_ptr<std::remove_pointer_t<glob_t>> glob_result(&raw_glob_result, &globfree);

    if (e != 0)
        throw std::runtime_error("Glob failed: " + s);

    for(std::size_t i = 0; i < glob_result->gl_pathc; ++i)
        result.push_back(glob_result->gl_pathv[i]);
    return result;
}

template<class T>
auto& deref(T* x)
{
    if (x == nullptr)
        throw std::runtime_error("Dereference null pointer");
    return *x;
}

std::string library_path()
{
    std::string path = "";
    Dl_info info;

    // Find the location of .so
    if(dladdr((void*)miopen_tensile_gemm_hip, &info))
    {
        path = info.dli_fname;
        auto i = path.rfind('/');
        if (i != std::string::npos)
            path = path.substr(0, i);
        else
            path = "";
    }
    return path + "/miopentensile/library/";
}

auto create_library()
{
    return Tensile::LoadLibraryFile<Tensile::ContractionProblem>(library_path() + "TensileLibrary.yaml");
    // return Tensile::EmbeddedLibrary<Tensile::ContractionProblem>::NewLibrary("miopen_tensile_kernels");
}

const auto& library()
{
    static auto result = create_library();
    assert(result != nullptr);
    return *result;
}

auto create_adaptor() {
    // Workaround: The Tensile::hip::SolutionAdapter is not a regular type, so heap allocate it instead
    auto a = std::make_shared<Tensile::hip::SolutionAdapter>();
    for(auto&& f:glob_files(library_path() + "*co"))
        a->loadCodeObjectFile(f);
    return a;
}

auto& adaptor()
{
    static auto result = create_adaptor();
    return *result;
}


bool is_transposed(const miopen_tensile_matrix& a)
{
    return a.strides[1] > a.strides[0];
}

size_t get_idx(const miopen_tensile_matrix& a, size_t n)
{
    return (n + (is_transposed(a) ? 1 : 0)) % 2;
}

size_t get_ld(const miopen_tensile_matrix& a)
{
    return a.strides[get_idx(a, 0)];
}

Tensile::DataType get_data_type(const miopen_tensile_matrix&)
{
    return Tensile::DataType::Float;
}

miopen_tensile_matrix transpose(const miopen_tensile_matrix& a)
{
    return miopen_tensile_matrix{{a.lens[1], a.lens[0]}, {a.strides[1], a.strides[0]}};
}

Tensile::ContractionProblem create_tensile_problem(const miopen_tensile_matrix& a, const miopen_tensile_matrix& b, const miopen_tensile_matrix& c)
{
    if (a.batch.num > 1 or b.batch.num > 1 or c.batch.num > 1)
    {
        auto batch = std::max({a.batch.num, b.batch.num, c.batch.num});
        return Tensile::ContractionProblem::GEMM_Strides(is_transposed(a), 
                                                         is_transposed(b), 
                                                         get_data_type(a), 
                                                         get_data_type(b), 
                                                         get_data_type(c), 
                                                         get_data_type(c), 
                                                         is_transposed(a) ? a.lens[0] : a.lens[1], 
                                                         is_transposed(b) ? b.lens[1] : b.lens[0], 
                                                         is_transposed(a) ? a.lens[1] : a.lens[0],
                                                         batch, 
                                                         get_ld(a),
                                                         a.batch.stride, 
                                                         get_ld(b),
                                                         b.batch.stride, 
                                                         get_ld(c),
                                                         c.batch.stride,
                                                         get_ld(c),
                                                         c.batch.stride,
                                                         1.0);
    }
    else
        return Tensile::ContractionProblem::GEMM(is_transposed(a),
                                                 is_transposed(b), 
                                                 is_transposed(a) ? a.lens[0] : a.lens[1], 
                                                 is_transposed(b) ? b.lens[1] : b.lens[0], 
                                                 is_transposed(a) ? a.lens[1] : a.lens[0], 
                                                 get_ld(a), 
                                                 get_ld(b), 
                                                 get_ld(c), 
                                                 1.0, 
                                                 false, 
                                                 1);
}

extern "C" {

miopen_tensile_status miopen_tensile_gemm_hip(hipStream_t stream, 
                                              miopen_tensile_matrix* a, 
                                              miopen_tensile_matrix* b, 
                                              miopen_tensile_matrix* c, 
                                              double alpha, 
                                              double beta)
{

    miopen_tensile_status status = miopen_tensile_status_unknown;
    std::shared_ptr<Tensile::ContractionSolution> solution;

        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
        std::shared_ptr<Tensile::Hardware>                                           hardware;
        Tensile::hip::SolutionAdapter                                                adapter;

hardware = Tensile::hip::GetCurrentDevice();

std::string dir = library_path() + "*co";
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }

library = std::dynamic_pointer_cast<
                Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
                Tensile::LoadLibraryFile<Tensile::ContractionProblem>(library_path() + "TensileLibrary.yaml"));


        static constexpr Tensile::DataType Tensile_Ti = Tensile::DataType::Float;
        static constexpr Tensile::DataType Tensile_To = Tensile::DataType::Float;

        // Tensor descriptors for ta, tb
        Tensile::TensorDescriptor ta, tb;

        // Tensor ops for matrices, like complex conjugate
        Tensile::TensorOps aops, bops, cops, dops;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblem::FreeIndices  freeIndex(2);
        Tensile::ContractionProblem::BoundIndices boundIndex(1);
        Tensile::ContractionProblem::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // Tensile does not support 0-sized dimensions. For when k == 0, we still need to
        // multiply C by beta, but not add any of the rank-0 dot products. As a workaround,
        // we pass k = 1 and set alpha == 0, since alpha == 0 has the same effect as k == 0.
        auto k = is_transposed(a) ? a.lens[1] : a.lens[0];
        k = k == 0 ? 1 : k;

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(is_transposed(a))
        {
            ta = {
                    Tensile_Ti,
                    {k, a.lens[0], a.batch.num},
                    {1, get_ld(a), a.batch.stride}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            ta = {
                    Tensile_Ti,
                    {a.lens[1], k, a.batch.num},
                    {1, get_ld(a), a.batch.stride}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(is_transposed(b))
        {
            tb = {
                    Tensile_Ti,
                    {b.lens[1], k, b.batch.num},
                    {1, get_ld(b), b.batch.stride}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            tb = {
                    Tensile_Ti,
                    {k, b.lens[0], b.batch.num},
                    {1, get_ld(b), b.batch.stride}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }


        // Descriptor for input matrix C
        Tensile::TensorDescriptor tc{Tensile_To,
                                    {is_transposed(a) ? a.lens[0] : a.lens[1], is_transposed(b) ? b.lens[1] : b.lens[0], c.batch.num},
                                    {1, get_ld(c), c.batch.stride}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor td{Tensile_To,
                                    {is_transposed(a) ? a.lens[0] : a.lens[1], is_transposed(b) ? b.lens[1] : b.lens[0], c.batch.num},
                                    {1, get_ld(c), c.batch.stride}};

        // The ContractionProblem
        Tensile::ContractionProblem tensileProblem{ta,
                                                   aops,
                                                   tb,
                                                   bops,
                                                   tc,
                                                   cops,
                                                   td,
                                                   dops,
                                                   freeIndex,
                                                   batchIndex,
                                                   boundIndex,
                                                   beta};

        // If HPA is active, mark it as true
        if(sizeof(Tc) > sizeof(Ti))
            tensileProblem.setHighPrecisionAccumulate(true);





        solution          = library->findBestSolution(tensileProblem, *hardware);

    Tensile::TypedContractionInputs<float> inputs;
    inputs.a = reinterpret_cast<const float*>(b->data);
    inputs.b = reinterpret_cast<const float*>(a->data);
    inputs.c = reinterpret_cast<const float*>(c->data);
    inputs.d = reinterpret_cast<float*>(c->data);
    inputs.alpha = alpha;
    inputs.beta = beta;

        if(!solution)
        {
            td::cerr << "No solution found." << std::endl;
            status          = miopen_tensile_status_no_solution;
        }
        else
        {
            auto handle = prob.handle;
            adapter.launchKernels(
                solution->solve(tensileProblem, inputs, *hardware),
                stream, nullptr, nullptr);
            status = rocblas_status_success;
        }

    return status;
}

}
