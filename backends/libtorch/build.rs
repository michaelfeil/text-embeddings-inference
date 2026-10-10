use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=../candle/extensions/candle-layer-norm/kernels");
    println!("cargo:rerun-if-changed=../candle/src/kernels/qwen35_gdn.cu");
    println!("cargo:rerun-if-changed=../candle/src/kernels/qk_norm_rope.cu");
    println!("cargo:rerun-if-changed=../candle/src/kernels/gemma_rms_norm.cu");
    println!("cargo:rerun-if-changed=../candle/src/kernels/gated_activation.cu");
    println!("cargo:rerun-if-changed=../candle/src/pooling_kernels/mean_pool.cu");
    let torch = PathBuf::from(env::var_os("LIBTORCH").expect(
        "Set LIBTORCH to the root of a LibTorch 2.14.1 C++ distribution (include/, lib/, share/)",
    ));
    let dst = cmake::Config::new("cpp")
        .define("CMAKE_PREFIX_PATH", &torch)
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=tei_torch");
    println!("cargo:lib_dir={}/lib", dst.display());
    // The bridge carries an rpath to LibTorch. Executables also need to find the bridge.
    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", dst.display());
    }
}
