#pragma once
#include "model.h"

namespace tei {
std::unique_ptr<Model> create_encoder(const Options&, Weights&, c10::Device, at::ScalarType);
}
