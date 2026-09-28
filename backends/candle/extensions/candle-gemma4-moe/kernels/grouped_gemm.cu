// SPDX-License-Identifier: MIT
// BF16 routed expert GEMM. All problem sizes, matrix pointers and strides live
// on the GPU; the grouped scheduler does not read routing counts on the host.
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>

using Element = cutlass::bfloat16_t;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;

extern "C" int gemma4_grouped_gemm_bf16(
    void *problem_sizes, int groups, void *a_ptrs, void *b_ptrs, void *c_ptrs,
    void *lda, void *ldb, void *ldc, cudaStream_t stream) {
    if (groups <= 0) return int(cutlass::Status::kErrorInvalidProblem);
    const int blocks = Gemm::sufficient();
    if (blocks <= 0) return int(cutlass::Status::kErrorInternal);
    typename Gemm::Arguments args(
        static_cast<cutlass::gemm::GemmCoord *>(problem_sizes), groups, blocks,
        {1.f, 0.f}, static_cast<Element **>(a_ptrs), static_cast<Element **>(b_ptrs),
        static_cast<Element **>(c_ptrs), static_cast<Element **>(c_ptrs),
        static_cast<int64_t *>(lda), static_cast<int64_t *>(ldb),
        static_cast<int64_t *>(ldc), static_cast<int64_t *>(ldc));
    Gemm op;
    auto status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return int(status);
    return int(op.run(stream));
}
