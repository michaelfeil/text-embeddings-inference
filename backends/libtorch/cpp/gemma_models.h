// Native adaptations of the Candle Gemma encoders. Upstream notices are retained
// in the repository LICENSE and NOTICE files.
#pragma once
#include "model.h"
namespace tei {
std::unique_ptr<Model> create_gemma(const Options&, Weights&, c10::Device, at::ScalarType);
}
