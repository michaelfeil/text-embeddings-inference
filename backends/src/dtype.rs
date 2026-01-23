use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum DType {
    // Float16 is not available on accelerate
    #[cfg(any(
        feature = "python",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
    Float32,
    #[cfg(feature = "python")]
    Bfloat16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Float16 is not available on accelerate
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),
            #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
            DType::Float32 => write!(f, "float32"),
            #[cfg(feature = "python")]
            DType::Bfloat16 => write!(f, "bfloat16"),

            // Handle cases where variants are not available
            #[cfg(not(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            )))]
            _ => write!(f, "unknown"),
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        #[cfg(any(feature = "accelerate", feature = "mkl", feature = "ort"))]
        {
            DType::Float32
        }
        #[cfg(not(any(
            feature = "accelerate",
            feature = "mkl",
            feature = "ort",
            feature = "python"
        )))]
        {
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            {
                DType::Float16
            }
            #[cfg(not(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            )))]
            {
                // Fallback - this should not happen in normal usage
                panic!("No valid DType available for current feature set");
            }
        }
        #[cfg(feature = "python")]
        {
            DType::Bfloat16
        }
    }
}
