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

    friend std::ostream& operator<<(std::ostream& os, const shape& x)
    {
        os << "{" << to_string_range(x.lens.begin(), x.lens.end()) << "}, ";
        os << "{" << to_string_range(x.strides.begin(), x.strides.end()) << "}";
        return os;
    }

};

shape create_mat_shape(std::size_t x, std::size_t y, bool transposed = false)
{
    if (transposed)
        return {{x, y}, {1, y}};
    else
        return {{x, y}, {y, 1}};
}

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
    gemm(as.lens[0], bs.lens[1], as.lens[1], shape_with(as, a), shape_with(bs, b), shape_with(cs, c));
    return c;
}

template<class Ptr>
miopen_tensile_matrix to_tensile_matrix(shape s, const Ptr& p)
{
    return miopen_tensile_matrix{{s.lens[0], s.lens[1]}, {s.strides[0], s.strides[1]}, p.get()};
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
    miopen_tensile_gemm(stream.get(), &am, &bm, &cm);
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

TEST_CASE(gemm1)
{
    verify_gemm<float>(create_mat_shape(2, 2, true), create_mat_shape(2, 2), create_mat_shape(2, 2));
}


} // namespace mitensile

int main(int argc, const char* argv[]) { test::run(argc, argv); }

