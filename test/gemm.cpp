#include <miopentensile/gemm.h>
#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <half.hpp>
#include "hip.hpp"
#include "array.hpp"
#include "test.hpp"

namespace mitensile {


using int8x4 = array<std::int8_t, 4>;
using doublex4 = array<double, 4>;
using half = half_float::half;

// Multidimensional for loop
inline auto dfor()
{
    return [](auto f) { f(); };
}

template <class T, class... Ts>
auto dfor(T x, Ts... xs)
{
    return [=](auto f) {
        for(T i = 0; i < x; i++)
        {
            dfor(xs...)([&](Ts... is) { f(i, is...); });
        }
    };
}

template<class T>
T increment(T start, int max)
{
    return T(start > max ? 1 : start + 1);
}

int8x4 increment(int8x4 start, int max)
{
    int8x4 result = start;
    for(std::size_t i = 0;i < 4;i++)
    {
        result[i] = start[i] > max ? 1 : start[i] + 1;
    }
    return result;
}

template<class T, class U>
std::vector<T> generate(std::size_t sz, U pstart)
{
    auto start = T(pstart);
    std::vector<T> result(sz);
    std::generate(result.begin(), result.end(), [&] {
        T r = start;
        start = increment(start, 6);
        return r;
    });
    return result;
}

template<class T, class U>
std::vector<T> fill(std::size_t sz, U pvalue)
{
    auto value = T(pvalue);
    std::vector<T> result(sz);
    std::fill(result.begin(), result.end(), value);
    return result;
}

template <class Iterator>
inline std::string to_string_range(Iterator start, Iterator last)
{
    std::stringstream ss;
    if(start != last)
    {
        ss << *start;
        std::for_each(std::next(start), last, [&](auto&& x) { ss << ", " << x; });
    }
    return ss.str();
}

struct shape
{
    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;
    void calculate_strides()
    {
        strides.clear();
        strides.resize(lens.size(), 0);
        if(strides.empty())
            return;
        strides.back() = 1;
        std::partial_sum(lens.rbegin(),
                         lens.rend() - 1,
                         strides.rbegin() + 1,
                         std::multiplies<std::size_t>());
    }

    std::size_t element_space() const
    {
        assert(lens.size() == strides.size());
        if(lens.empty())
            return 0;
        return std::inner_product(lens.begin(),
                                  lens.end(),
                                  strides.begin(),
                                  std::size_t{0},
                                  std::plus<std::size_t>{},
                                  [](std::size_t l, std::size_t s) { return (l - 1) * s; }) +
               1;
    }

    std::size_t elements() const
    {
        assert(lens.size() == strides.size());
        if(lens.empty())
            return 0;
        return std::accumulate(
            lens.begin(), lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }

    std::size_t index(const std::vector<std::size_t>& l) const
    {
        assert(l.size() <= this->lens.size());
        assert(this->lens.size() == this->strides.size());
        return std::inner_product(l.begin(), l.end(), this->strides.begin(), std::size_t{0});
    }
    std::size_t index_ik(std::vector<std::size_t> l, std::size_t k) const
    {
        l.back() = k;
        return this->index(l);
    }

    std::size_t index_kj(std::vector<std::size_t> l, std::size_t k) const
    {
        l.at(l.size() - 2) = k;
        return this->index(l);
    }

    shape transpose() const
    {
        assert(lens.size() == strides.size());
        shape r = *this;
        if (r.lens.size() > 1) 
        {
            std::reverse(r.lens.end()-2, r.lens.end());
            std::reverse(r.strides.end()-2, r.strides.end());
        }
        return r;
    }

    shape step(int axis, int step) const
    {
        shape r = *this;
        r.lens[axis] /= step;
        if (r.strides[axis] == *std::max_element(r.strides.begin(), r.strides.end()))
            return r;
        std::set<std::size_t, std::greater<std::size_t>> sorted_strides(r.strides.begin(), r.strides.end());
        sorted_strides.erase(sorted_strides.find(r.strides[axis]), sorted_strides.end());
        assert(not sorted_strides.empty());
        auto stride = *std::prev(sorted_strides.end());
        auto it = std::find(r.strides.begin(), r.strides.end(), stride);
        assert(it != r.strides.end());
        *it /= step;
        return r;
    }

    template<class T>
    std::vector<T> generate(std::size_t seed=0) const
    {
        return mitensile::generate<T>(element_space(), seed);
    }

    template<class T>
    std::vector<T> fill(std::size_t x=0) const
    {
        return mitensile::fill<T>(element_space(), x);
    }

    static shape from_lens(std::vector<std::size_t> l)
    {
        shape r;
        r.lens = l;
        r.calculate_strides();
        return r;
    }

    template <class F>
    void for_each(F f)
    {
        assert(lens.size() == strides.size());
        // Ensure calls to f use const ref to vector
        auto call = [&f](const std::vector<std::size_t>& i) { f(i); };
        std::vector<std::size_t> indices(lens.size());
        shape ss = from_lens(lens);
        for(std::size_t i = 0; i < ss.elements(); i++)
        {
            std::transform(ss.strides.begin(),
                           ss.strides.end(),
                           ss.lens.begin(),
                           indices.begin(),
                           [&](std::size_t stride, std::size_t len) {
                               assert(len > 0 and stride > 0);
                               return (i / stride) % len;
                           });
            call(indices);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const shape& x)
    {
        os << "{" << to_string_range(x.lens.begin(), x.lens.end()) << "}, ";
        os << "{" << to_string_range(x.strides.begin(), x.strides.end()) << "}";
        return os;
    }

};

template<class R>
auto shape_with(const shape& s, R&& x)
{
    return [&](std::size_t i, std::size_t j) -> auto& {
        return x[s.index({i, j})];
    };
}

template<class T, class Out = T>
struct problem
{
    static problem generate(shape as, shape bs, shape cs)
    {
        problem result;
        result.as = as;
        result.bs = bs;
        result.cs = cs;
        result.a = as.generate<T>(1);
        result.b = bs.generate<T>(2);
        result.c = cs.fill<Out>(0);
        return result;
    }
    shape as;
    shape bs;
    shape cs;
    std::vector<T> a;
    std::vector<T> b;
    std::vector<Out> c;
};

template<class T>
T accumulate(T x)
{
    return x;
}

template<class T, std::size_t N>
T accumulate(const array<T, N>& x)
{
    return x.sum();
}

template<class T, class Out = T>
std::vector<Out> cpu_gemm(problem<T, Out> p)
{
    auto k = p.as.lens.back();
    p.cs.for_each([&](auto idx) {
        double x = 0.0;
        dfor(k)([&](int kk) { 
            // x += a(i, kk) * b(kk, j); 
            x += accumulate(p.a[p.as.index_ik(idx, kk)] * p.b[p.bs.index_kj(idx, kk)]); 
        });
        p.c[p.cs.index(idx)] = x;
    });
    return p.c;
}

template<miopen_tensile_type N>
using tensile_type_const = std::integral_constant<miopen_tensile_type, N>;

template<class T>
struct get_data_type {};

template<>
struct get_data_type<float> : tensile_type_const<miopen_tensile_type_float>
{};

template<>
struct get_data_type<half> : tensile_type_const<miopen_tensile_type_half>
{};

template<>
struct get_data_type<int8x4> : tensile_type_const<miopen_tensile_type_int8x4>
{};

template<>
struct get_data_type<std::int32_t> : tensile_type_const<miopen_tensile_type_int32>
{};

template<class T, class Ptr>
miopen_tensile_matrix to_tensile_matrix(shape s, const Ptr& p)
{
    if (s.lens.size() == 2)
        return miopen_tensile_matrix{{s.lens[0], s.lens[1]}, {s.strides[0], s.strides[1]}, {0, 0}, get_data_type<T>{}, p.get()};
    else if (s.lens.size() == 3)
        return miopen_tensile_matrix{{s.lens[1], s.lens[2]}, {s.strides[1], s.strides[2]}, {s.lens[0], s.strides[0]}, get_data_type<T>{}, p.get()};
    else
        throw std::runtime_error("Invalid shape to to_tensile_matrix");
}

template<class T, class Out = T>
std::vector<Out> gpu_gemm(const problem<T, Out>& p)
{
    auto a = to_gpu(p.a);
    auto b = to_gpu(p.b);
    auto c = to_gpu(p.c);
    auto am = to_tensile_matrix<T>(p.as, a);
    auto bm = to_tensile_matrix<T>(p.bs, b);
    auto cm = to_tensile_matrix<Out>(p.cs, c);

    auto stream = create_stream();
    auto e = miopen_tensile_gemm_hip(stream.get(), &am, &bm, &cm, 1.0, 0.0);
    if (e != miopen_tensile_status_success)
        throw std::runtime_error("Failed to run miopen_tensile_gemm_hip");
    auto r = from_gpu<Out>(cm.data, p.cs.element_space());
    return r;
}

template<class T, class Out = T>
void verify_gemm(shape as, shape bs, shape cs)
{
    std::cout << "a -> " << as << std::endl;
    std::cout << "b -> " << bs << std::endl;
    std::cout << "c -> " << cs << std::endl;
    auto p = problem<T, Out>::generate(as, bs, cs);
    auto cpu = cpu_gemm(p);
    auto gpu = gpu_gemm(p);
    EXPECT(cpu == gpu);
}
void verify_gemm(shape as, shape bs, shape cs)
{
    verify_gemm<float>(as, bs, cs);
    verify_gemm<half>(as, bs, cs);
}
void verify_int8x4_gemm(shape as, shape bs, shape cs)
{
    std::cout << "a -> " << as << std::endl;
    std::cout << "b -> " << bs << std::endl;
    std::cout << "c -> " << cs << std::endl;
    auto p = problem<int8x4, std::int32_t>::generate(as.step(1, 4), bs.step(0, 4), cs);
    auto gpu_p = p;
    gpu_p.as = as;
    gpu_p.bs = bs;
    auto cpu = cpu_gemm(p);
    auto gpu = gpu_gemm(gpu_p);
    EXPECT(cpu == gpu);
}

shape create_mat_shape(std::vector<std::size_t> l, bool transposed = false)
{
    auto s = shape::from_lens(l);
    if (transposed)
        return s.transpose();
    else
        return s;
}

TEST_CASE(gemm1)
{
    verify_gemm(create_mat_shape({2, 2}, true),
                       create_mat_shape({2, 2}), 
                       create_mat_shape({2, 2}));
}

TEST_CASE(gemm12)
{
    verify_gemm(create_mat_shape({4, 1}, true),
                       create_mat_shape({4, 1}), 
                       create_mat_shape({1, 1}));
}

TEST_CASE(gemm2)
{
    verify_gemm(create_mat_shape({8, 4}),
                       create_mat_shape({4, 32}), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm31)
{
    verify_gemm(create_mat_shape({8, 4}),
                       create_mat_shape({4, 32}), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm3)
{
    verify_gemm(create_mat_shape({4, 8}, true),
                       create_mat_shape({4, 32}), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm4)
{
    verify_gemm(create_mat_shape({8, 4}),
                       create_mat_shape({32, 4}, true), 
                       create_mat_shape({8, 32}));
}

TEST_CASE(bgemm1)
{
    verify_gemm(create_mat_shape({2, 2, 2}),
                       create_mat_shape({2, 2, 2}), 
                       create_mat_shape({2, 2, 2}));
}
TEST_CASE(bgemm21)
{
    verify_gemm(create_mat_shape({3, 3, 2}),
                       create_mat_shape({3, 2, 4}),
                       create_mat_shape({3, 3, 4}));
}
TEST_CASE(bgemm2)
{
    verify_gemm(create_mat_shape({64, 8, 4}),
                       create_mat_shape({64, 4, 32}),
                       create_mat_shape({64, 8, 32}));
}
TEST_CASE(bgemm3)
{
    verify_gemm(create_mat_shape({64, 4, 8}, true),
                       create_mat_shape({64, 32, 4}, true),
                       create_mat_shape({64, 8, 32}));
}

TEST_CASE(int8gemm1)
{
    verify_int8x4_gemm(create_mat_shape({2, 4}),
                       create_mat_shape({4, 2}), 
                       create_mat_shape({2, 2}));
}

TEST_CASE(int8gemm2)
{
    verify_int8x4_gemm(create_mat_shape({3, 4}),
                       create_mat_shape({4, 5}), 
                       create_mat_shape({3, 5}));
}

TEST_CASE(int8bgemm1)
{
    verify_int8x4_gemm(create_mat_shape({32, 2, 4}),
                       create_mat_shape({32, 4, 2}), 
                       create_mat_shape({32, 2, 2}));
}

TEST_CASE(large_gemm1)
{
    verify_gemm<float>(create_mat_shape({1024, 1024}),
                       create_mat_shape({1024, 1024}), 
                       create_mat_shape({1024, 1024}));
}
TEST_CASE(large_gemm2)
{
    verify_gemm<float>(create_mat_shape({1024, 2048},true),
                       create_mat_shape({2048, 1024},true),
                       create_mat_shape({2048, 2048}));
}

} // namespace mitensile

int main(int argc, const char* argv[]) { test::run(argc, argv); }

