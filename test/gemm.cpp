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
std::vector<T> rand_generate(std::size_t sz, T scale = 1, long long seed = 0)
{
    std::vector<T> result(sz);
    srand(seed);
    for(auto itr = result.begin(); itr < result.end() ; itr++ )
        *itr = static_cast<T>(int(double(scale)*(rand() / static_cast<double>(RAND_MAX))));
//    printf("\n");
//    for(auto i : result)
//        printf("rand_gene  %f\n", float(i));
//    printf("\n");
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
std::vector<T> cpu_gemm(miopen_tensile_matrix as, miopen_tensile_matrix bs, miopen_tensile_matrix cs, std::vector<T> va, std::vector<T> vb, std::vector<T> vc)
{
    auto c = vc;
    auto m = cs.is_mat_transposed ? cs.lens[1] : cs.lens[0];
    auto n = cs.is_mat_transposed ? cs.lens[0] : cs.lens[1];
    auto k = as.is_mat_transposed ? as.lens[0] : as.lens[1];
    for(auto idx = 0; idx < vc.size(); idx++) 
    {
        auto cbi = idx / cs.strides[0] / cs.strides[1] / cs.lens[0];
	auto cmi = (idx / cs.strides[0] / cs.strides[1]) % cs.lens[0];
        auto cni = idx % (cs.strides[0] * cs.strides[1]);
	if(cs.is_mat_transposed)
	    std::swap(cmi, cni);
	if(cmi < m && cni < n)
	{
            double x = 0.0;
            dfor(k)([&](int kk) {
                int idx_a = cbi * as.strides[0] * as.strides[1] * as.lens[0] + cmi * as.strides[0] + kk * as.strides[1];
		int idx_b = cbi * bs.strides[0] * bs.strides[1] * bs.lens[0] + kk * bs.strides[0] + cni * bs.strides[1];
                x += va[idx_a] * vb[idx_b]; 
            });
            c[idx] += x;
	}
    }
//    for(auto i : c)
//        printf("cpu c mat : %f\n", float(i));
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

std::size_t get_mat_size(miopen_tensile_matrix mat)
{
    return mat.strides[0] * mat.strides[1] * mat.lens[0] * std::max(mat.batch.num, std::size_t(1));
}

template<class T>
std::vector<T> gpu_gemm(miopen_tensile_matrix as, miopen_tensile_matrix bs, miopen_tensile_matrix cs, std::vector<T> va, std::vector<T> vb, std::vector<T> vc)
{
    auto a = to_gpu(va);
    auto b = to_gpu(vb);
    auto c = to_gpu(vc);
    as.data = a.get();
    bs.data = b.get();
    cs.data = c.get();

    auto stream = create_stream();
    auto e = miopen_tensile_gemm_hip(stream.get(), &as, &bs, &cs, 1.0, 0.0);
    if (e != miopen_tensile_status_success)
        throw std::runtime_error("Failed to run miopen_tensile_gemm_hip");
    auto r = from_gpu<T>(cs.data, get_mat_size(cs));
//    for(auto i : r)
//        printf("gpu c mat : %f\n", float(i));
    return r;
}

template<class T>
void verify_gemm(miopen_tensile_matrix as, miopen_tensile_matrix bs, miopen_tensile_matrix cs)
{
    std::cout << "a -> " << as.lens[0] << " " << as.lens[1] << std::endl;
    std::cout << "b -> " << bs.lens[0] << " " << bs.lens[1] << std::endl;
    std::cout << "c -> " << cs.lens[0] << " " << cs.lens[1] << std::endl;

    auto va = rand_generate<T>(get_mat_size(as), T(32), 0);
    auto vb = rand_generate<T>(get_mat_size(bs), T(32), 0xFF);
    auto vc = fill<T>(get_mat_size(cs), T(0));

    auto cpu = cpu_gemm<T>(as, bs, cs, va, vb, vc);
    auto gpu = gpu_gemm<T>(as, bs, cs, va, vb, vc);
    EXPECT(cpu == gpu);
}

std::size_t get_stride(std::vector<std::size_t> l, int idx, bool transposed = false)
{
    return (idx == 0 && transposed) || (idx == 1 && !transposed) ? 1 : *(l.rbegin());
}

std::size_t get_batch_stride(std::vector<std::size_t> l)
{
    	return l.size() == 3 ? std::accumulate(l.rbegin().base() - 2, l.rbegin().base(), std::size_t{1}, std::multiplies<std::size_t>()) : 0;
}

miopen_tensile_matrix init_mat(std::vector<std::size_t> l, bool transposed = false, miopen_tensile_type dtype = miopen_tensile_type_float)
{
    miopen_tensile_matrix s{{*(l.rbegin() + 1), *(l.rbegin())},
                              {get_stride(l, 0, transposed), get_stride(l, 1, transposed)},
                              {l.size() == 3 ? *(l.begin()) : 1, get_batch_stride(l)},
                              dtype,
                              transposed,
                              nullptr};

    printf("mat lens               :  %f   %f\n", float(s.lens[0]), float(s.lens[1]));
    printf("mat strides            :  %f   %f\n", float(s.strides[0]), float(s.strides[1]));
    printf("mat batch size, stride :  %f   %f\n", float(s.batch.num), float(s.batch.stride));
    printf("mat transpose          :  %d\n", int(s.is_mat_transposed));

    return s;
}

TEST_CASE(gemm1)
{
    verify_gemm<float>(init_mat({2, 2}, true),
                       init_mat({2, 2}), 
                       init_mat({2, 2}));
}

TEST_CASE(gemm2)
{
    verify_gemm<float>(init_mat({8, 4}),
                       init_mat({4, 32}), 
                       init_mat({8, 32}));
}
TEST_CASE(gemm3)
{
    verify_gemm<float>(init_mat({4, 8}, true),
                       init_mat({4, 32}), 
                       init_mat({8, 32}));
}
TEST_CASE(gemm4)
{
    verify_gemm<float>(init_mat({8, 4}),
                       init_mat({32, 4}, true), 
                       init_mat({8, 32}));
}
TEST_CASE(gemm5)
{
    verify_gemm<float>(init_mat({1024, 1024}),
                       init_mat({1024, 1024}), 
                       init_mat({1024, 1024}));
}
TEST_CASE(gemm6)
{
    verify_gemm<float>(init_mat({1024, 2048},true),
                       init_mat({2048, 1024},true),
                       init_mat({2048, 2048}));
}

TEST_CASE(bgemm1)
{
    verify_gemm<float>(init_mat({2, 2, 2}),
                       init_mat({2, 2, 2}), 
                       init_mat({2, 2, 2}));
}
TEST_CASE(bgemm2)
{
    verify_gemm<float>(init_mat({64, 8, 4}),
                       init_mat({64, 4, 32}),
                       init_mat({64, 8, 32}));
}
TEST_CASE(bgemm3)
{
    verify_gemm<float>(init_mat({64, 4, 8}, true),
                       init_mat({64, 32, 4}, true),
                       init_mat({64, 8, 32}));
}
} // namespace mitensile

int main(int argc, const char* argv[]) { test::run(argc, argv); }

