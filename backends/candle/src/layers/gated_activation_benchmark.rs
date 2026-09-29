//! Opt-in CUDA-event timing of the existing approximate GELU gate.
use super::*;
use candle::cuda_backend::cudarc::driver::sys::CUevent_flags;

#[test]
#[ignore = "requires an idle CUDA GPU; prints p50/p95 microseconds"]
fn packed_glu_latency() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    for dtype in [DType::F16, DType::BF16] {
        for (batch, tokens) in [(1, 128), (8, 512), (32, 1024)] {
            let input =
                Tensor::randn(0f32, 2f32, (batch, tokens, 5248), &device)?.to_dtype(dtype)?;
            let run = |fused| -> Result<Tensor> {
                if fused {
                    gated_activation(&input, Some(&HiddenAct::Gelu))
                } else {
                    let chunks = input.chunk(2, candle::D::Minus1)?;
                    chunks[0].gelu()?.mul(&chunks[1])
                }
            };
            super::tests::compare_one(&input, true)?;
            for _ in 0..10 {
                let _ = run(false)?;
                let _ = run(true)?;
            }
            device.synchronize()?;
            let mut samples = [vec![], vec![]];
            // Alternate order; each event sample averages ten invocations.
            for sample in 0..40 {
                for fused in [sample % 2 == 0, sample % 2 != 0] {
                    let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                    for _ in 0..10 {
                        let _ = run(fused)?;
                    }
                    let stop = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                    samples[usize::from(fused)].push(start.elapsed_ms(&stop)? as f64 * 100.);
                }
            }
            for (fused, mut values) in samples.into_iter().enumerate() {
                values.sort_by(f64::total_cmp);
                eprintln!(
                    "{dtype:?} {batch}x{tokens}x2624 fused={fused} p50_us={:.3} p95_us={:.3}",
                    (values[19] + values[20]) / 2.,
                    values[37]
                );
            }
        }
    }
    Ok(())
}
