#ifndef MIOPENTENSILE_GUARD_VXD0PQ_ARRAY_HPP
#define MIOPENTENSILE_GUARD_VXD0PQ_ARRAY_HPP

#include <type_traits>

namespace mitensile {

#define MIOPENTENSILE_ARRAY_OP(op, binary_op)                                                    \
    array& operator op(const array& x)                           \
    {                                                                                              \
        for(std::size_t i = 0; i < N; i++)                                                           \
            d[i] op x[i];                                                                          \
        return *this;                                                                              \
    }                                                                                              \
    array& operator op(const T& x)                                   \
    {                                                                                              \
        for(std::size_t i = 0; i < N; i++)                                                           \
            d[i] op x;                                                                             \
        return *this;                                                                              \
    }                                                                                              \
    friend array operator binary_op(array x, const array& y) \
    {                                                                                              \
        return x op y;                                                                             \
    }                                                                                              \
    friend array operator binary_op(array x, const T& y)         \
    {                                                                                              \
        return x op y;                                                                             \
    }                                                                                              \
    friend array operator binary_op(const T& y, array x)         \
    {                                                                                              \
        return x op y;                                                                             \
    }

template <class T, std::size_t N>
struct array
{
    T d[N];
    array() = default;
    array(const array&) = default;
    array& operator=(const array&) = default;
    array(T x)
    {
        for(std::size_t i=0;i<N;i++)
            d[i] = x;
    }
    template<class U, class = typename std::enable_if<not std::is_same<T, U>{} and std::is_convertible<U, T>{}>::type>
    array(const array<U, N>& x)
    {
        for(std::size_t i=0;i<N;i++)
            d[i] = x.d[i];
    }
    T& operator[](std::size_t i) { return d[i]; }
    const T& operator[](std::size_t i) const { return d[i]; }

    T& front() { return d[0]; }
    const T& front() const { return d[0]; }

    T& back() { return d[N - 1]; }
    const T& back() const { return d[N - 1]; }

    T* data() { return d; }
    const T* data() const { return d; }

    std::integral_constant<std::size_t, N> size() const { return {}; }

    T* begin() { return d; }
    const T* begin() const { return d; }

    T* end() { return d + size(); }
    const T* end() const { return d + size(); }

    T sum() const
    {
        T result = 0;
        for(std::size_t i = 0; i < N; i++)
            result += d[i];
        return result;
    }

    MIOPENTENSILE_ARRAY_OP(+=, +)
    MIOPENTENSILE_ARRAY_OP(*=, *)
    MIOPENTENSILE_ARRAY_OP(/=, /)
    MIOPENTENSILE_ARRAY_OP(%=, %)
    MIOPENTENSILE_ARRAY_OP(&=, &)
    MIOPENTENSILE_ARRAY_OP(|=, |)
    MIOPENTENSILE_ARRAY_OP(^=, ^)

    friend bool operator==(const array& x, const array& y)
    {
        for(std::size_t i = 0; i < N; i++)
        {
            if(x[i] != y[i])
                return false;
        }
        return true;
    }

    friend bool operator!=(const array& x, const array& y)
    {
        return !(x == y);
    }

    template<class Stream>
    friend Stream& operator<<(Stream& os, const array& x)
    {
        for(auto&& e:x)
            os << x << ", ";
        return os;
    }
};

} // namespace mitensile
#endif // MIOPENTENSILE_GUARD_VXD0PQ_ARRAY_HPP