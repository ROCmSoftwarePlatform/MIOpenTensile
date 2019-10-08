#ifndef MIOPEN_TENSILE_GUARD_HIP_HPP
#define MIOPEN_TENSILE_GUARD_HIP_HPP

#include "manage_ptr.hpp"
#include <hip/hip_runtime_api.h>
#include <stdexcept>

namespace mitensile {

using hip_ptr = MIOPEN_TENSILE_MANAGE_PTR(void, hipFree);

inline std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

inline std::size_t get_available_gpu_memory()
{
    size_t free;
    size_t total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        throw std::runtime_error("Failed getting available memory: " + hip_error(status));
    return free;
}

inline hip_ptr allocate_gpu(std::size_t sz, bool host = false)
{
    if(sz > get_available_gpu_memory())
        throw std::runtime_error("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = host ? hipHostMalloc(&result, sz) : hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        if(host)
            throw std::runtime_error("Gpu allocation failed: " + hip_error(status));
        else
            allocate_gpu(sz, true);
    }
    return hip_ptr{result};
}

template<class T>
inline hip_ptr allocate_gpu(std::size_t sz, bool host = false)
{
    return allocate_gpu(sz * sizeof(T), host);
}

inline hip_ptr write_to_gpu(const void* x, std::size_t sz, bool host = false)
{
    auto result = allocate_gpu(sz, host);
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        throw std::runtime_error("Copy to gpu failed: " + hip_error(status));
    return result;
}

template <class T>
std::vector<T> from_gpu(const void* x, std::size_t sz)
{
    std::vector<T> result(sz);
    auto status = hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        throw std::runtime_error("Copy from gpu failed: " + hip_error(status)); // NOLINT
    return result;
}

template <class T>
hip_ptr to_gpu(const T& x)
{
    using type = typename T::value_type;
    auto size  = x.size() * sizeof(type);
    return write_to_gpu(x.data(), size);
}

using hip_stream_ptr = MIOPEN_TENSILE_MANAGE_PTR(hipStream_t, hipStreamDestroy);

static hip_stream_ptr create_stream()
{
    hipStream_t result = nullptr;
    auto status        = hipStreamCreateWithFlags(&result, hipStreamNonBlocking);
    if(status != hipSuccess)
        throw std::runtime_error("Failed to allocate stream");
    return hip_stream_ptr{result};
}

} // namespace mitensile

#endif
