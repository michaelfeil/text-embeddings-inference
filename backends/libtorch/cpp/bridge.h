#pragma once
#include <stdint.h>
#include <stddef.h>

struct TeiConfig {
  int64_t hidden, heads, layers, intermediate, vocab, positions, types;
  double epsilon;
  int32_t activation;
};

// Every fallible entry point catches C++ exceptions; the error string is thread-local.
extern "C" {
const char* tei_error();
int64_t tei_device_count(const char* device);
void* tei_create(const TeiConfig*, const char* device, int32_t dtype);
int32_t tei_weight(void*, const char* name, const uint8_t* data,
                   const int64_t* shape, size_t rank, int32_t dtype);
int32_t tei_ready(void*);
int32_t tei_forward(void*, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output);
void tei_destroy(void*);
}
