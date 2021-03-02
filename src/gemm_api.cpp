#include <miopentensile/gemm.h>
#include <Tensile/Tensile.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <dlfcn.h>
#include <glob.h>

#define MIOT_DEBUG_PRINTOUTS 0

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
    return Tensile::LoadLibraryFile<Tensile::ContractionProblem>(library_path() +
#if TENSILE_USE_LLVM && !TENSILE_USE_MSGPACK
        "TensileLibrary.yaml"
#else
        "TensileLibrary.dat"
#endif
        );
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

Tensile::DataType get_data_type(const miopen_tensile_matrix& a)
{
    switch(a.type)
    {
    case miopen_tensile_type_float: return Tensile::DataType::Float;
    case miopen_tensile_type_half: return Tensile::DataType::Half;
    case miopen_tensile_type_int8x4: return Tensile::DataType::Int8x4;
    case miopen_tensile_type_int32: return Tensile::DataType::Int32;
    case miopen_tensile_type_bfloat16: return Tensile::DataType::BFloat16;
    }
}

miopen_tensile_matrix transpose(const miopen_tensile_matrix& a)
{
    return miopen_tensile_matrix{{a.lens[1], a.lens[0]}, {a.strides[1], a.strides[0]}};
}

Tensile::ContractionProblem create_tensile_problem(const miopen_tensile_matrix& a, const miopen_tensile_matrix& b, const miopen_tensile_matrix& c)
{
    if (a.lens[0] != b.lens[1])
      throw std::runtime_error("K dimensions do not match");
    if (a.lens[1] != c.lens[1])
      throw std::runtime_error("M dimensions do not match");
    if (b.lens[0] != c.lens[0])
      throw std::runtime_error("N dimensions do not match");

    if (a.batch.num > 1 or b.batch.num > 1 or c.batch.num > 1 or a.type != miopen_tensile_type_float or b.type != miopen_tensile_type_float or c.type != miopen_tensile_type_float)
    {
        auto batch = std::max({a.batch.num, b.batch.num, c.batch.num});
        auto k = a.lens[0];
        auto lda = get_ld(a);
        auto ldb = get_ld(b);
        auto stride_a = a.batch.stride;
        auto stride_b = b.batch.stride;
        if(a.type == miopen_tensile_type_int8x4)
        {
            if(k % 4 != 0 || (is_transposed(a) && (lda % 4 != 0))|| (!is_transposed(b) && (ldb % 4 != 0))|| (c.batch.num > 1 && (a.batch.stride % 4 != 0 || b.batch.stride % 4 != 0)))
            {
                std::cerr << "Invalid int8 problem size." << std::endl;
                return Tensile::ContractionProblem{};
            }
            else
            {
                k /= 4;
                lda = !is_transposed(a) ? lda : lda / 4;
                ldb = !is_transposed(b) ? ldb / 4 : ldb;
                stride_a /= 4;
                stride_b /= 4;
            }
        }
#if MIOT_DEBUG_PRINTOUTS
	printf("\n");
        printf("tensile gemm_strides\n");
        printf("is_transposed(a)  %d\n", int(is_transposed(a)));
        printf("is_transposed(b)  %d\n", int(is_transposed(b)));
        printf("is_transposed(c)  %d\n", int(is_transposed(c)));
        printf("a.lens[0]  %zu\n", a.lens[0]);
        printf("a.lens[1]  %zu\n", a.lens[1]);
        printf("b.lens[0]  %zu\n", b.lens[0]);
        printf("b.lens[1]  %zu\n", b.lens[1]);
        printf("batch  %zu\n", batch);
        printf("get_ld(a)  %zu\n", get_ld(a));
        printf("a.batch.stride  %zu\n", a.batch.stride);
        printf("get_ld(b)  %zu\n", get_ld(b));
        printf("b.batch.stride  %zu\n", b.batch.stride);
        printf("get_ld(c)  %zu\n", get_ld(c));
        printf("c.batch.stride  %zu\n", c.batch.stride);
        printf("\n"); 
	printf("c.lens[0]  %zu\n", c.lens[0]);
        printf("c.lens[1]  %zu\n", c.lens[1]);
        printf("M  %zu\n", a.lens[1]);
        printf("N  %zu\n", b.lens[0]);
        printf("K  %zu\n", a.lens[0]);
        printf("\n");
#endif

        auto problem = Tensile::ContractionProblem::GEMM_Strides(is_transposed(a), 
                                                                 is_transposed(b), 
                                                                 get_data_type(a), 
                                                                 get_data_type(b), 
                                                                 get_data_type(c), 
                                                                 get_data_type(c), 
                                                                 a.lens[1],
                                                                 b.lens[0],
                                                                 k,
                                                                 batch,
                                                                 lda,
                                                                 stride_a,
                                                                 ldb,
                                                                 stride_b,
                                                                 get_ld(c),
                                                                 c.batch.stride,
                                                                 get_ld(c),
                                                                 c.batch.stride,
                                                                 1.0);

        if (a.type == miopen_tensile_type_half || a.type == miopen_tensile_type_bfloat16 || a.type == miopen_tensile_type_int8x4)
            problem.setHighPrecisionAccumulate(true);

        return problem;
    }
    else
    {
#if MIOT_DEBUG_PRINTOUTS
        printf("\n");
        printf("tensile gemm\n");
        printf("is_transposed(a)  %d\n", int(is_transposed(a)));
        printf("is_transposed(b)  %d\n", int(is_transposed(b)));
        printf("is_transposed(c)  %d\n", int(is_transposed(c)));
        printf("a.lens[0]  %zu\n", a.lens[0]);
        printf("a.lens[1]  %zu\n", a.lens[1]);
        printf("b.lens[0]  %zu\n", b.lens[0]);
        printf("b.lens[1]  %zu\n", b.lens[1]);
        printf("get_ld(a)  %zu\n", get_ld(a));
        printf("a.batch.stride  %zu\n", a.batch.stride);
        printf("get_ld(b)  %zu\n", get_ld(b));
        printf("b.batch.stride  %zu\n", b.batch.stride);
        printf("get_ld(c)  %zu\n", get_ld(c));
        printf("c.batch.stride  %zu\n", c.batch.stride);
        printf("\n");
        printf("c.lens[0]  %zu\n", c.lens[0]);
        printf("c.lens[1]  %zu\n", c.lens[1]);
        printf("M  %zu\n", a.lens[1]);
        printf("N  %zu\n", b.lens[0]);
        printf("K  %zu\n", a.lens[0]);

        printf("\n");
#endif

        return Tensile::ContractionProblem::GEMM(is_transposed(a),
                                                 is_transposed(b), 
                                                 a.lens[1],
                                                 b.lens[0],
                                                 a.lens[0],
                                                 get_ld(a), 
                                                 get_ld(b), 
                                                 get_ld(c), 
                                                 1.0, 
                                                 false, 
                                                 1);
    }
}

template <typename A, typename B = A, typename C = A, typename D = C, typename Alpha = C, typename Beta = C>
miopen_tensile_status launch_kernels(hipStream_t& stream, 
                                     Tensile::ContractionProblem& problem, 
                                     std::shared_ptr<Tensile::Hardware>& hardware, 
                                     std::shared_ptr<Tensile::ContractionProblem::Solution>& solution, 
                                     miopen_tensile_matrix* a, 
                                     miopen_tensile_matrix* b, 
                                     miopen_tensile_matrix* c,
                                     double alpha, 
                                     double beta)
{
    Tensile::TypedContractionInputs<A, B, C, D, Alpha, Beta> inputs;
    inputs.a = reinterpret_cast<const A*>(b->data);
    inputs.b = reinterpret_cast<const B*>(a->data);
    inputs.c = reinterpret_cast<const C*>(c->data);
    inputs.d = reinterpret_cast<D*>(c->data);
    inputs.alpha = Alpha(alpha);
    inputs.beta = Beta(beta);
    auto kernels = solution->solve(problem, inputs, *hardware);
    adaptor().launchKernels(kernels, stream, nullptr, nullptr);
    return miopen_tensile_status_success;
}

extern "C" {

miopen_tensile_status miopen_tensile_gemm_hip(hipStream_t stream, 
                                              miopen_tensile_matrix* a, 
                                              miopen_tensile_matrix* b, 
                                              miopen_tensile_matrix* c, 
                                              double alpha, 
                                              double beta)
{
    auto problem = create_tensile_problem(deref(b), deref(a), deref(c));
    auto hardware = Tensile::hip::GetCurrentDevice();
    auto solution = library().findBestSolution(problem, *hardware);
    if (not solution)
    {
        std::cerr << "No solution found." << std::endl;
        return miopen_tensile_status_no_solution;
    }
    switch(a->type)
    {
    case miopen_tensile_type_float:
        return launch_kernels<float>(stream, problem, hardware, solution, a, b, c, alpha, beta);
    case miopen_tensile_type_half:
        return launch_kernels<Tensile::Half>(stream, problem, hardware, solution, a, b, c, alpha, beta);
    case miopen_tensile_type_int8x4:
        return launch_kernels<Tensile::Int8x4, Tensile::Int8x4, int32_t>(stream, problem, hardware, solution, a, b, c, alpha, beta);
    case miopen_tensile_type_int32:
        return miopen_tensile_status_no_solution;
    case miopen_tensile_type_bfloat16:
        return launch_kernels<Tensile::BFloat16, Tensile::BFloat16, Tensile::BFloat16, Tensile::BFloat16, float, float>(stream, problem, hardware, solution, a, b, c, alpha, beta);
    }
}

}
