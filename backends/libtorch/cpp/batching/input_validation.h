// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "bridge.h"
#include <c10/util/Exception.h>
#include <algorithm>
#include <limits>

namespace tei {
inline size_t validate_output_indices(const int32_t* cumulative,int64_t batch,
    const int64_t* pooled,size_t pooled_count,const int64_t* raw,size_t raw_count) {
  TORCH_CHECK(!pooled_count||pooled,"Missing counted pooled indices");
  TORCH_CHECK(!raw_count||raw,"Missing counted raw indices");
  TORCH_CHECK(pooled_count<=std::numeric_limits<size_t>::max()/sizeof(int64_t)&&
    raw_count<=std::numeric_limits<size_t>::max()/sizeof(int64_t),"Overflowing output index count");
  for(size_t i=0;i<pooled_count;++i)
    TORCH_CHECK(pooled[i]>=0&&pooled[i]<batch,"Invalid pooled output index");
  size_t rows=0;
  for(size_t i=0;i<raw_count;++i) {
    TORCH_CHECK(raw[i]>=0&&raw[i]<batch,"Invalid raw output index");
    const size_t length=int64_t(cumulative[raw[i]+1])-cumulative[raw[i]];
    TORCH_CHECK(length<=std::numeric_limits<size_t>::max()-rows,"Raw output row count overflow");
    rows+=length;
  }
  return rows;
}
inline size_t expected_output_elements(size_t pooled_count,size_t raw_rows,
    int64_t pooled_width,int64_t raw_width) {
  auto product=[](size_t count,int64_t width) {
    if(!count)return size_t(0);
    TORCH_CHECK(width>0&&static_cast<uint64_t>(width)<=std::numeric_limits<size_t>::max()/count,
      "Native output size overflow");
    return count*static_cast<size_t>(width);
  };
  const auto pooled=product(pooled_count,pooled_width),raw=product(raw_rows,raw_width);
  TORCH_CHECK(raw<=std::numeric_limits<size_t>::max()-pooled,"Native output size overflow");
  const auto elements=pooled+raw;
  TORCH_CHECK(elements<=std::numeric_limits<size_t>::max()/sizeof(float),"Native output byte size overflow");
  return elements;
}
inline int64_t input_product(int64_t left,int64_t right,const char* name) {
  TORCH_CHECK(left>0&&right>0&&left<=std::numeric_limits<int64_t>::max()/right,
    "Invalid or overflowing ",name," dimensions");
  return left*right;
}
inline void input_bytes(int64_t elements,size_t element_size,const char* name) {
  TORCH_CHECK(elements>0&&static_cast<uint64_t>(elements)<=
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max())/element_size&&
    static_cast<uint64_t>(elements)<=std::numeric_limits<size_t>::max()/element_size,
    "Overflowing ",name," byte size");
}
// Validate descriptors without constructing tensors or reading media payloads.
// This is structural admission; a request memory budget is a separate layer.
inline int64_t validate_input_descriptors(const int64_t* ids,const int64_t* types,
    const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
    const int64_t* media_positions,const TeiImage* images,size_t image_count,
    const TeiAudio* audios,size_t audio_count,bool single_frame_images=false) {
  TORCH_CHECK(batch>0&&batch<std::numeric_limits<int64_t>::max()&&cumulative,
    "Invalid native packed batch");
  input_bytes(batch+1,sizeof(int32_t),"cumulative offsets");
  TORCH_CHECK(cumulative[0]==0&&max_sequence>0,"Invalid native packed sequence geometry");
  for(int64_t row=0;row<batch;++row) {
    const int64_t start=cumulative[row],end=cumulative[row+1];
    TORCH_CHECK(start>=0&&end>start&&end-start<=max_sequence,
      "Native packed sequences must be nonempty and fit max_sequence");
  }
  const int64_t tokens=cumulative[batch];
  TORCH_CHECK(max_sequence<=tokens,"Native maximum sequence exceeds packed token count");
  input_bytes(tokens,sizeof(int64_t),"token buffer");
  TORCH_CHECK(ids&&types&&positions,"Missing native token buffer");
  if(media_positions)input_bytes(input_product(3,tokens,"media positions"),sizeof(int64_t),"media positions");
  TORCH_CHECK(!image_count||images,"Missing counted image descriptors");
  TORCH_CHECK(!audio_count||audios,"Missing counted audio descriptors");
  TORCH_CHECK(image_count<=std::numeric_limits<size_t>::max()/sizeof(TeiImage)&&
    audio_count<=std::numeric_limits<size_t>::max()/sizeof(TeiAudio),"Overflowing media descriptor count");
  auto span=[&](int64_t start,int64_t count,const char* name) {
    TORCH_CHECK(start>=0&&start<tokens&&count>0&&count<=tokens-start,
      "Invalid ",name," token span");
    const auto* end=std::upper_bound(cumulative,cumulative+batch+1,start);
    TORCH_CHECK(count<=int64_t(*end)-start,"Media token span crosses packed sequence boundary");
    return int64_t(*(end-1));
  };
  for(size_t i=0;i<image_count;++i) {
    const auto& item=images[i];
    TORCH_CHECK(item.pixels&&item.merge_size>0,"Invalid image payload or merge size");
    const auto rows=input_product(input_product(item.grid[0],item.grid[1],"image grid"),item.grid[2],"image grid");
    TORCH_CHECK(!single_frame_images||item.grid[0]==1,"Gemma image grid must have one frame");
    TORCH_CHECK(item.rows==rows&&item.grid[1]%item.merge_size==0&&item.grid[2]%item.merge_size==0,
      "Image rows or merge geometry do not match grid");
    input_bytes(input_product(item.rows,item.patch_dim,"image payload"),sizeof(float),"image payload");
    const auto merged_rows=input_product(input_product(item.grid[0],item.grid[1]/item.merge_size,"merged image grid"),item.grid[2]/item.merge_size,"merged image grid");
    TORCH_CHECK(item.token_count==merged_rows,"Image token count does not match merged grid");
    const auto sequence_start=span(item.token_start,item.token_count,"image");
    TORCH_CHECK(item.sequence_start==sequence_start,
      "Image sequence start does not match containing packed sequence");
  }
  for(size_t i=0;i<audio_count;++i) {
    const auto& item=audios[i];
    TORCH_CHECK(item.values&&item.mask,"Missing audio features or validity mask");
    input_bytes(input_product(item.frames,item.feature_size,"audio payload"),sizeof(float),"audio payload");
    input_bytes(item.frames,sizeof(uint8_t),"audio validity mask");
    span(item.token_start,item.token_count,"audio");
  }
  return tokens;
}
}
