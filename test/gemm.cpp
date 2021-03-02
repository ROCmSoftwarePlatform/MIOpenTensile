#include <miopentensile/gemm.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include "hip.hpp"
#include "test.hpp"

namespace mitensile {

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

template <class AF, class BF, class CF>
void gemm(std::size_t n, std::size_t m, std::size_t k, AF a, BF b, CF c)
{
    dfor(n, m)([&](int i, int j) {
        double x = 0.0;
        dfor(k)([&](int kk) { x += a(i, kk) * b(kk, j); });
        c(i, j) = x;
    });
}

template<class T>
std::vector<T> generate(std::size_t sz, T start)
{
    std::vector<T> result(sz);
    std::iota(result.begin(), result.end(), start);
    return result;
}

template<class T>
std::vector<T> fill(std::size_t sz, T value)
{
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

template<class T>
std::vector<T> cpu_gemm(shape as, shape bs, shape cs)
{
    auto a = as.generate<T>(1);
    auto b = bs.generate<T>(2);
    auto c = cs.fill<T>(0);
    auto k = as.lens.back();
    cs.for_each([&](auto idx) {
        double x = 0.0;
        dfor(k)([&](int kk) { 
            // x += a(i, kk) * b(kk, j); 
            x += a[as.index_ik(idx, kk)] * b[bs.index_kj(idx, kk)]; 
        });
        c[cs.index(idx)] = x;
    });
    return c;
}

template<class Ptr>
miopen_tensile_matrix to_tensile_matrix(shape s, const Ptr& p)
{
    if (s.lens.size() == 2)
        return miopen_tensile_matrix{{s.lens[0], s.lens[1]}, {s.strides[0], s.strides[1]}, {0, 0}, miopen_tensile_type_float, p.get()};
    else if (s.lens.size() == 3)
        return miopen_tensile_matrix{{s.lens[1], s.lens[2]}, {s.strides[1], s.strides[2]}, {s.lens[0], s.strides[0]}, miopen_tensile_type_float, p.get()};
    else
        throw std::runtime_error("Invalid shape to to_tensile_matrix");
}

template<class T>
std::vector<T> gpu_gemm(shape as, shape bs, shape cs)
{
    auto a = to_gpu(as.generate<T>(1));
    auto b = to_gpu(bs.generate<T>(2));
    auto c = to_gpu(cs.fill<T>(0));
    auto am = to_tensile_matrix(as, a);
    auto bm = to_tensile_matrix(bs, b);
    auto cm = to_tensile_matrix(cs, c);

    auto stream = create_stream();
    auto e = miopen_tensile_gemm_hip(stream.get(), &am, &bm, &cm, 1.0, 0.0);
    if (e != miopen_tensile_status_success)
        throw std::runtime_error("Failed to run miopen_tensile_gemm_hip");
    auto r = from_gpu<T>(cm.data, cs.element_space());
    return r;
}

template<class T>
void verify_gemm(shape as, shape bs, shape cs)
{
    std::cout << "a -> " << as << std::endl;
    std::cout << "b -> " << bs << std::endl;
    std::cout << "c -> " << cs << std::endl;
    auto cpu = cpu_gemm<T>(as, bs, cs);
    auto gpu = gpu_gemm<T>(as, bs, cs);
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
    verify_gemm<float>(create_mat_shape({2, 2}, true),
                       create_mat_shape({2, 2}), 
                       create_mat_shape({2, 2}));
}

TEST_CASE(gemm12)
{
    verify_gemm<float>(create_mat_shape({4, 1}, true),
                       create_mat_shape({4, 1}), 
                       create_mat_shape({1, 4}));
}

TEST_CASE(gemm2)
{
    verify_gemm<float>(create_mat_shape({8, 4}),
                       create_mat_shape({4, 32}), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm3)
{
    verify_gemm<float>(create_mat_shape({4, 8}, true),
                       create_mat_shape({4, 32}), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm4)
{
    verify_gemm<float>(create_mat_shape({8, 4}),
                       create_mat_shape({32, 4}, true), 
                       create_mat_shape({8, 32}));
}
TEST_CASE(gemm5)
{
    verify_gemm<float>(create_mat_shape({1024, 1024}),
                       create_mat_shape({1024, 1024}), 
                       create_mat_shape({1024, 1024}));
}
TEST_CASE(gemm6)
{
    verify_gemm<float>(create_mat_shape({1024, 2048},true),
                       create_mat_shape({2048, 1024},true),
                       create_mat_shape({2048, 2048}));
}

TEST_CASE(bgemm1)
{
    verify_gemm<float>(create_mat_shape({2, 2, 2}),
                       create_mat_shape({2, 2, 2}), 
                       create_mat_shape({2, 2, 2}));
}
TEST_CASE(bgemm2)
{
    verify_gemm<float>(create_mat_shape({64, 8, 4}),
                       create_mat_shape({64, 4, 32}),
                       create_mat_shape({64, 8, 32}));
}
TEST_CASE(bgemm3)
{
    verify_gemm<float>(create_mat_shape({64, 4, 8}, true),
                       create_mat_shape({64, 32, 4}, true),
                       create_mat_shape({64, 8, 32}));
}


} // namespace mitensile

int main(int argc, const char* argv[]) { test::run(argc, argv); }

