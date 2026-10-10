use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-changed=cpp");
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
