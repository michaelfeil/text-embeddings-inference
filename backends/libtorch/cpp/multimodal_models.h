#pragma once
#include "model.h"
namespace tei {
std::unique_ptr<Model> create_multimodal(const Options&, Weights&, c10::Device, at::ScalarType);
}
