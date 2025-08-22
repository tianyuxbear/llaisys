#include "tensor.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

size_t Tensor::dim(size_t i) const {
    return _meta.shape[i];
}

ptrdiff_t Tensor::stride(size_t i) const {
    return _meta.strides[i];
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    auto m_shape = _meta.shape;
    auto m_strides = _meta.strides;

    size_t ndim_ = m_shape.size();
    std::vector<ptrdiff_t> c_strides(ndim_);
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        if (m_strides[ndim_ - i] != stride) {
            return false;
        }
        stride *= m_shape[ndim_ - i];
    }

    return true;
}

tensor_t Tensor::dimMerge(size_t dim_start, size_t dim_end) const {
    CHECK_ARGUMENT(dim_start <= dim_end && dim_end < ndim(), "dim_start and dim_end must be in right range");

    size_t new_ndim = ndim() - (dim_end - dim_start);
    std::vector<size_t> new_shape(new_ndim);
    std::vector<ptrdiff_t> new_strides(new_ndim);
    size_t index = 0;

    for (size_t i = 0; i < dim_start; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    new_shape[index] = 1;
    for (size_t i = dim_start; i <= dim_end; i++) {
        new_shape[index] *= dim(i);
    }

    new_strides[index] = stride(dim_end);
    index++;

    for (size_t i = dim_end + 1; i < ndim(); i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::dimSplit(size_t axis, const std::vector<size_t> &dims) const {
    size_t ndim_ = ndim();

    CHECK_ARGUMENT(dim(axis) == std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>()), "axis and dims must be in right range");

    size_t new_ndim = ndim_ + dims.size() - 1;
    std::vector<size_t> new_shape(new_ndim);
    std::vector<ptrdiff_t> new_strides(new_ndim);
    size_t index = 0;
    for (size_t i = 0; i < axis; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape[index] = dims[i];
        new_strides[index] = stride(axis) * dim(axis) / std::accumulate(dims.begin(), dims.begin() + i + 1, (size_t)1, std::multiplies<size_t>());
        index++;
    }
    for (size_t i = axis + 1; i < ndim_; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    auto ndim_ = ndim();
    CHECK_ARGUMENT(order.size() == ndim_, "order.size() must be equal to ndim");
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        CHECK_ARGUMENT(std::find(order.begin(), order.end(), i) != order.end(), "permute is given a bad order");
        new_shape[i] = dim(order[i]);
        new_strides[i] = stride(order[i]);
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // Check contiguous and shape compatibility
    if (!this->isContiguous()) {
        throw std::runtime_error("view() requires contiguous tensor");
    }
    size_t new_numel = 1;
    for (auto dim : shape) {
        new_numel *= dim;
    }
    if (new_numel != this->numel()) {
        throw std::runtime_error("view() shape is incompatible with number of elements");
    }
    // Construct new tensor meta
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }
    TensorMeta new_meta = this->_meta;
    new_meta.shape = shape;
    new_meta.strides = new_strides;
    // Create new tensor, share storage
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim() && start <= end && end <= _meta.shape[dim], "slice: bad arguments");
    const size_t slice_size = end - start;

    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = slice_size;

    std::vector<ptrdiff_t> new_strides = _meta.strides;
    const size_t offset = static_cast<size_t>(_meta.strides[dim] * start * utils::dsize(_meta.dtype));

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, offset));
}

void Tensor::load(const void *src_) {
    llaisysDeviceType_t device_type = deviceType();
    int device_id = deviceId();

    if (device_type != core::context().runtime().deviceType() || device_id != core::context().runtime().deviceId()) {
        core::context().setDevice(device_type, device_id);
    }
    core::context().runtime().api()->memcpy_sync(_storage->memory(), src_, _storage->size(), LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
