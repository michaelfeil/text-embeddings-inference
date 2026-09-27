//! Experimental row-scaled E4M3 GEMM. Not connected to serving paths.
use candle::{DType, Result, Storage, Tensor};
use cudarc::cublaslt::{result as lt, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::f16;
use std::mem::size_of;

fn err(e: lt::CublasError) -> candle::Error {
    candle::Error::Cuda(Box::new(e))
}
macro_rules! owned {
    ($name:ident, $ty:ty, $destroy:path) => {
        struct $name($ty);
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    let _ = $destroy(self.0);
                }
            }
        }
    };
}
owned!(Handle, sys::cublasLtHandle_t, lt::destroy_handle);
owned!(Desc, sys::cublasLtMatmulDesc_t, lt::destroy_matmul_desc);
owned!(
    Layout,
    sys::cublasLtMatrixLayout_t,
    lt::destroy_matrix_layout
);
owned!(
    Preference,
    sys::cublasLtMatmulPreference_t,
    lt::destroy_matmul_pref
);
impl Desc {
    // Only used with the attribute's documented scalar or device-pointer type.
    fn set<T>(&self, attr: sys::cublasLtMatmulDescAttributes_t, value: &T) -> Result<()> {
        unsafe {
            lt::set_matmul_desc_attribute(
                self.0,
                attr,
                value as *const T as *const _,
                size_of::<T>(),
            )
            .map_err(err)
        }
    }
}

struct Plan {
    desc: Desc,
    a: Layout,
    b: Layout,
    c: Layout,
    algo: sys::cublasLtMatmulHeuristicResult_t,
}

/// Experimental GEMM executor tied to one Candle CUDA device/stream.
/// Exclusive mutable access serializes host submission and workspace reuse.
/// No unsafe Send/Sync implementation: create/use it on its owning worker thread.
pub struct Fp8Matmul {
    handle: Handle,
    plans: std::collections::HashMap<(usize, usize, usize), Plan>,
    workspace: cudarc::driver::CudaSlice<u8>,
    device: candle::Device,
}
impl Fp8Matmul {
    pub fn new(device: &candle::Device) -> Result<Self> {
        let dev = device.as_cuda_device()?;
        dev.cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)?;
        let handle = Handle(lt::create_handle().map_err(err)?);
        let workspace = unsafe { dev.alloc::<u8>(32usize << 20)? };
        Ok(Self {
            handle,
            plans: Default::default(),
            workspace,
            device: device.clone(),
        })
    }
    /// Compute FP16 `(x * sx) @ (w * sw).T` from contiguous E4M3 matrices.
    /// Scales are FP32, one per row. Plans are bounded to sixteen shapes and
    /// never retain input pointers between submissions without refreshing them.
    pub fn scaled_mm(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        sx: &Tensor,
        sw: &Tensor,
    ) -> Result<Tensor> {
        if !self.device.same_device(x.device()) {
            candle::bail!("FP8 executor belongs to a different CUDA device")
        }
        let (m, k) = x.dims2()?;
        let (n, wk) = w.dims2()?;
        if m == 0 || n == 0 || k == 0 || wk != k || k % 16 != 0 || n % 16 != 0 {
            candle::bail!("FP8 GEMM requires nonzero matching dimensions and N/K multiples of 16")
        }
        // cuBLAS uses signed dimensions internally even though layout construction is u64.
        for d in [m, n, k] {
            i32::try_from(d).map_err(candle::Error::wrap)?;
        }
        let count = m
            .checked_mul(n)
            .ok_or_else(|| candle::Error::Msg("FP8 output size overflow".into()))?;
        for (t, dt, len) in [
            (x, DType::F8E4M3, m * k),
            (w, DType::F8E4M3, n * k),
            (sx, DType::F32, m),
            (sw, DType::F32, n),
        ] {
            if t.dtype() != dt
                || !t.is_contiguous()
                || t.elem_count() != len
                || !t.device().same_device(x.device())
            {
                candle::bail!("FP8 matrices/scales must have matching CUDA devices, contiguous layouts and expected types/shapes")
            }
            if t.storage_and_layout().1.start_offset() != 0 {
                candle::bail!("FP8 prototype requires zero-offset tensors")
            }
        }
        let dev = x.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)?;
        let (xs, _) = x.storage_and_layout();
        let (ws, _) = w.storage_and_layout();
        let (ssx, _) = sx.storage_and_layout();
        let (ssw, _) = sw.storage_and_layout();
        let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ssx), Storage::Cuda(ssw)) =
            (&*xs, &*ws, &*ssx, &*ssw)
        else {
            candle::bail!("FP8 GEMM requires CUDA")
        };
        let xa = xs.as_cuda_slice::<float8::F8E4M3>()?;
        let wa = ws.as_cuda_slice::<float8::F8E4M3>()?;
        let xscale = ssx.as_cuda_slice::<f32>()?;
        let wscale = ssw.as_cuda_slice::<f32>()?;
        let mut output = unsafe { dev.alloc::<f16>(count)? };
        let workspace_bytes = 32usize << 20;
        // Keep all pointer records alive until after enqueueing the GEMM. This is
        // essential for producer-stream waits and safe asynchronous buffer reuse.
        let (xp, _rx) = xa.device_ptr(&stream);
        let (wp, _rw) = wa.device_ptr(&stream);
        let (spx, _rsx) = xscale.device_ptr(&stream);
        let (spw, _rsw) = wscale.device_ptr(&stream);
        let (op, ro) = output.device_ptr_mut(&stream);
        let (scratch, rwork) = self.workspace.device_ptr_mut(&stream);
        let key = (m, n, k);
        if !self.plans.contains_key(&key) {
            if self.plans.len() >= 16 {
                self.plans.clear();
            }
            let d = Desc(
                lt::create_matmul_desc(
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType::CUDA_R_32F,
                )
                .map_err(err)?,
            );
            use sys::cublasLtMatmulDescAttributes_t::*;
            d.set(
                CUBLASLT_MATMUL_DESC_TRANSA,
                &cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            )?;
            d.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &1i8)?;
            let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            d.set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode)?;
            d.set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode)?;
            d.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &spw)?;
            d.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &spx)?;
            let a = Layout(
                lt::create_matrix_layout(
                    sys::cudaDataType::CUDA_R_8F_E4M3,
                    k as u64,
                    n as u64,
                    k as i64,
                )
                .map_err(err)?,
            );
            let b = Layout(
                lt::create_matrix_layout(
                    sys::cudaDataType::CUDA_R_8F_E4M3,
                    k as u64,
                    m as u64,
                    k as i64,
                )
                .map_err(err)?,
            );
            let c = Layout(
                lt::create_matrix_layout(
                    sys::cudaDataType::CUDA_R_16F,
                    n as u64,
                    m as u64,
                    n as i64,
                )
                .map_err(err)?,
            );
            let pref = Preference(lt::create_matmul_pref().map_err(err)?);
            unsafe {
                lt::set_matmul_pref_attribute(
            pref.0,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes as *const _ as *const _,
            size_of::<usize>(),
        )
        .map_err(err)?;
            }
            let algo = unsafe {
                lt::get_matmul_algo_heuristic(self.handle.0, d.0, a.0, b.0, c.0, c.0, pref.0)
                    .map_err(err)?
            };
            self.plans.insert(
                key,
                Plan {
                    desc: d,
                    a,
                    b,
                    c,
                    algo,
                },
            );
        }
        let plan = self.plans.get(&key).expect("plan inserted above");
        // Scale allocations may change even when dimensions do not.
        plan.desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &spw,
        )?;
        plan.desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &spx,
        )?;
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            lt::matmul(
                self.handle.0,
                plan.desc.0,
                &alpha as *const _ as *const _,
                &beta as *const _ as *const _,
                wp as *const _,
                plan.a.0,
                xp as *const _,
                plan.b.0,
                op as *const _,
                plan.c.0,
                op as *mut _,
                plan.c.0,
                &plan.algo.algo,
                scratch as *mut _,
                workspace_bytes,
                stream.cu_stream() as *mut _,
            )
            .map_err(err)?;
        }
        drop(ro);
        drop(rwork);
        let storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        Ok(Tensor::from((
            Storage::Cuda(storage),
            candle::Shape::from((m, n)),
        )))
    }
}
/// Convenience entry point; inference should retain `Fp8Matmul` across calls.
pub fn scaled_mm(x: &Tensor, w: &Tensor, sx: &Tensor, sw: &Tensor) -> Result<Tensor> {
    Fp8Matmul::new(x.device())?.scaled_mm(x, w, sx, sw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_scales_and_irregular_m() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        for m in [1, 17, 128, 17, 1] {
            let x = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); m * 32], (m, 32), &dev)?;
            let w = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); 16 * 32], (16, 32), &dev)?;
            let sx = Tensor::from_vec(
                (0..m).map(|i| (i + 1) as f32 / 32.).collect::<Vec<_>>(),
                m,
                &dev,
            )?;
            let sw = Tensor::from_vec(
                (0..16).map(|i| (i + 1) as f32 / 16.).collect::<Vec<_>>(),
                16,
                &dev,
            )?;
            let y = executor
                .scaled_mm(&x, &w, &sx, &sw)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            for i in 0..m {
                for j in 0..16 {
                    assert_eq!(y[i][j], (i + 1) as f32 * (j + 1) as f32 / 16.);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn signed_values_and_layout_guards() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let (m, n, k) = (17, 32, 64);
        let xv = (0..m * k)
            .map(|i| ((i * 7 % 9) as f32 - 4.) / 4.)
            .collect::<Vec<_>>();
        let wv = (0..n * k)
            .map(|i| ((i * 11 % 13) as f32 - 6.) / 4.)
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(
            xv.iter()
                .copied()
                .map(float8::F8E4M3::from_f32)
                .collect::<Vec<_>>(),
            (m, k),
            &dev,
        )?;
        let w = Tensor::from_vec(
            wv.iter()
                .copied()
                .map(float8::F8E4M3::from_f32)
                .collect::<Vec<_>>(),
            (n, k),
            &dev,
        )?;
        let sx = Tensor::full(0.5f32, m, &dev)?;
        let sw = Tensor::full(0.25f32, n, &dev)?;
        let y = scaled_mm(&x, &w, &sx, &sw)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        for i in 0..m {
            for j in 0..n {
                let reference = (0..k).map(|q| xv[i * k + q] * wv[j * k + q]).sum::<f32>() * 0.125;
                assert_eq!(y[i][j], reference);
            }
        }
        assert!(scaled_mm(&x, &w, &sx.narrow(0, 0, m - 1)?, &sw).is_err());
        assert!(scaled_mm(&x.t()?, &w, &sx, &sw).is_err());
        assert!(scaled_mm(&x.narrow(0, 1, m - 1)?, &w, &sx.narrow(0, 1, m - 1)?, &sw).is_err());
        assert!(scaled_mm(&Tensor::ones((m, k), DType::F16, &dev)?, &w, &sx, &sw).is_err());
        Ok(())
    }
    #[test]
    fn cache_refreshes_scales_and_is_bounded() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        let w = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); 16 * 32], (16, 32), &dev)?;
        let sw = Tensor::full(0.25f32, 16, &dev)?;
        let mut pending = Vec::new();
        for m in 1..=17 {
            let x = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); m * 32], (m, 32), &dev)?;
            let sa = Tensor::full(0.5f32, m, &dev)?;
            let sb = Tensor::full(0.25f32, m, &dev)?;
            for (scale, expected) in [(&sa, 4f32), (&sb, 2f32), (&sa, 4f32)] {
                pending.push((executor.scaled_mm(&x, &w, scale, &sw)?, expected));
            }
            assert!(executor.plans.len() <= 16);
        }
        assert_eq!(executor.plans.len(), 1);
        // Read only after cache eviction and after temporary scale/input owners
        // have been dropped, exercising asynchronous lifetime tracking.
        for (result, expected) in pending {
            let result = result
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(result.iter().all(|v| *v == expected));
        }
        Ok(())
    }
}
