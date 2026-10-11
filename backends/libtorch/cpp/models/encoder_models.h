#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
std::unique_ptr<Model> create_encoder(const Options&, Weights&, c10::Device, at::ScalarType);
}
