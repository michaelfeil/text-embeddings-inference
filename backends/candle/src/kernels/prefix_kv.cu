// Copy cached prefix and suffix rows into reusable packed KV storage. Half values
// are copied as bits, preserving FP16/BF16 exactly. One block handles one row.
extern "C" __global__ void prefix_kv_copy(
    unsigned short* dst, const unsigned short* saved,
    const unsigned short* suffix, const unsigned int* rows,
    unsigned int saved_rows, unsigned int width,
    unsigned long long saved_stride, unsigned long long suffix_stride) {
    unsigned int row = rows[blockIdx.x];
    const unsigned short* src = row < saved_rows
        ? saved + row * saved_stride
        : suffix + (row - saved_rows) * suffix_stride;
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x)
        dst[(unsigned long long)blockIdx.x * width + col] = src[col];
}

// Every destination appears once; source storage is separate from the cache.
extern "C" __global__ void prefix_kv_scatter(
    unsigned short* dst, const unsigned short* src, const unsigned int* pairs,
    unsigned int width, unsigned long long dst_stride,
    unsigned long long src_stride) {
    unsigned int dest = pairs[2 * blockIdx.x], source = pairs[2 * blockIdx.x + 1];
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x)
        dst[dest * dst_stride + col] = src[source * src_stride + col];
}
