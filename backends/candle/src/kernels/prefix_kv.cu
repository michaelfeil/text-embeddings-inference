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
