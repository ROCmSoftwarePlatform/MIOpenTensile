#include <miopentensile/gemm.h>
#include <algorithm>
#include <numeric>
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

template<class F>
auto flip(F f)
{
    return [=](auto x, auto y) {
        return f(y, x);
    };
}

template <class T>
auto with_stride(T& data, std::size_t stride)
{
    return [&data, stride](auto x, auto y) -> auto& {
        return data.at(x* stride + y);
    };
}

template <class T>
auto with_stride(T* data, std::size_t stride)
{
    return [&data, stride](auto x, auto y) -> auto& {
        return data[x* stride + y];
    };
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

template<class T>
std::vector<T> cpu_gemm(std::size_t n)
{
    auto a = generate<T>(n*n, 1);
    auto b = generate<T>(n*n, 2);
    auto c = generate<T>(n*n, 0);
    gemm(n, n, n, with_stride(a, n), with_stride(b, n), with_stride(c, n));
    return c;
}

template<class T>
std::vector<T> gpu_gemm(std::size_t n)
{
    auto create_matrix = [&](auto& x, bool transposed = false) {
        if (transposed)
            return miopen_tensile_matrix{{n, n}, {1, n}, x.get()};
        else
            return miopen_tensile_matrix{{n, n}, {n, 1}, x.get()};
    };
    auto a = to_gpu(generate<T>(n*n, 1));
    auto b = to_gpu(generate<T>(n*n, 2));
    auto c = to_gpu(fill<T>(n*n, 0));
    auto am = create_matrix(a);
    auto bm = create_matrix(b, true);
    auto cm = create_matrix(c);

    auto stream = create_stream();
    miopen_tensile_gemm(stream.get(), &am, &bm, &cm);
    auto r = from_gpu<T>(cm.data, n*n);
    return r;
}

TEST_CASE(gemm1)
{
    auto cpu = cpu_gemm<float>(4);
    auto gpu = gpu_gemm<float>(4);
    EXPECT(cpu == gpu);
}


} // namespace mitensile

int main(int argc, const char* argv[]) { test::run(argc, argv); }

