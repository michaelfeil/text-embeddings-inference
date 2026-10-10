#pragma once
#include <stdint.h>
#include <stddef.h>

struct TeiConfig {
  int64_t hidden, heads, layers, intermediate, vocab, positions, types;
  double epsilon;
  int32_t activation;
};

struct TeiImage {
  const float* pixels;
  int64_t rows, patch_dim, grid[3], merge_size, token_start, token_count, sequence_start;
};
struct TeiAudio {
  const float* values;
  const uint8_t* mask;
  int64_t frames, feature_size, token_start, token_count;
};
struct TeiDecisionField {
  int64_t kind, question_start, question_end;
  const int64_t* options;
  size_t option_count;
};
struct TeiDecision {
  int32_t kind;
  int64_t question_type;
  const int64_t* markers;
  size_t marker_count;
  const int64_t* token_ids;
  size_t token_count;
  const TeiDecisionField* fields;
  size_t field_count;
};

// Every fallible entry point catches C++ exceptions; the error string is thread-local.
extern "C" {
const char* tei_error();
int64_t tei_device_count(const char* device);
void* tei_create(const TeiConfig*, const char* device, int32_t dtype);
int32_t tei_weight(void*, const char* name, const uint8_t* data,
                   const int64_t* shape, size_t rank, int32_t dtype);
int32_t tei_option(void*, const char* key, const char* value);
int64_t tei_output_width(void*);
int64_t tei_pooled_width(void*);
int64_t tei_classification_width(void*);
int64_t tei_graph_count(void*);
int32_t tei_ready(void*);
int32_t tei_decision_counts(void*,const TeiDecision*,size_t count,int64_t* counts);
int32_t tei_decide(void*,const int64_t* ids,const int64_t* types,
  const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
  const TeiDecision*,float* logits,size_t capacity,float* actions,
  const int64_t* media_positions,const TeiImage* images,size_t image_count,
  const TeiAudio* audios,size_t audio_count);
int32_t tei_forward(void*, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count);
void tei_destroy(void*);
}
