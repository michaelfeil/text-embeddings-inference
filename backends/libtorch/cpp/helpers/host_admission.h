// Contribution terms: see LICENSE and LICENSING.md at the repository root.
#pragma once
#include "helpers/device_admission.h"
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace tei {
inline std::shared_ptr<DeviceAdmissionState> host_admission_state() {
  static auto state=std::make_shared<DeviceAdmissionState>();return state;
}
inline bool host_controller(const std::string& list,const std::string& controller) {
  return (","+list+",").find(","+controller+",")!=std::string::npos;
}
inline std::optional<size_t> host_memory_number(const std::string& text) {
  if(text=="max")return {};
  size_t value=0;const auto result=std::from_chars(text.data(),text.data()+text.size(),value);
  require_memory_admission(result.ec==std::errc{}&&result.ptr==text.data()+text.size(),
    "Native host memory accounting is invalid");
  return value;
}
inline std::string host_mount_path(std::string text) {
  for(size_t i=0;i+3<text.size();++i)if(text[i]=='\\'&&
      text[i+1]>='0'&&text[i+1]<='7'&&text[i+2]>='0'&&text[i+2]<='7'&&text[i+3]>='0'&&text[i+3]<='7') {
    const char value=(text[i+1]-'0')*64+(text[i+2]-'0')*8+text[i+3]-'0';
    text.replace(i,4,1,value);
  }
  return text;
}
// Query on each admission: cgroup usage and membership can change while serving.
inline size_t host_memory_available(const std::filesystem::path& proc="/proc") {
  namespace fs=std::filesystem;
  std::ifstream meminfo(proc/"meminfo");std::string line;std::optional<size_t> available;
  while(std::getline(meminfo,line)) {
    std::istringstream fields(line);std::string key,value,unit;fields>>key>>value>>unit;
    if(key=="MemAvailable:") {
      const auto kib=host_memory_number(value);
      require_memory_admission(kib&&unit=="kB"&&*kib<=std::numeric_limits<size_t>::max()/1024,
        "Native host available-memory accounting is unavailable");
      available=*kib*1024;break;
    }
  }
  require_memory_admission(available.has_value(),"Native host available-memory accounting is unavailable");
  std::ifstream groups(proc/"self/cgroup");std::string membership;bool unified=true;
  require_memory_admission(groups.is_open(),"Native host cgroup membership is unavailable");
  while(std::getline(groups,line)) {
    const auto first=line.find(':'),second=line.find(':',first==std::string::npos?first:first+1);
    if(second==std::string::npos)continue;
    const auto controllers=line.substr(first+1,second-first-1);
    if(host_controller(controllers,"memory")){membership=line.substr(second+1);unified=false;break;}
    if(first==1&&line[0]=='0'&&controllers.empty())membership=line.substr(second+1);
  }
  if(membership.empty())return *available; // No memory cgroup controller.
  std::ifstream mounts(proc/"self/mountinfo");
  while(std::getline(mounts,line)) {
    const auto separator=line.find(" - ");if(separator==std::string::npos)continue;
    std::istringstream tail(line.substr(separator+3));std::string type,source,options;tail>>type>>source>>options;
    if(unified?type!="cgroup2":type!="cgroup"||!host_controller(options,"memory"))continue;
    std::istringstream fields(line.substr(0,separator));std::string id,parent,device,root,mount;
    fields>>id>>parent>>device>>root>>mount;
    fs::path boundary=host_mount_path(mount);
    if(boundary.has_relative_path()&&boundary.filename().empty())boundary=boundary.parent_path();
    auto relative=fs::path(membership).lexically_relative(host_mount_path(root));
    require_memory_admission(!relative.empty()&&*relative.begin()!="..",
      "Native host cgroup memory hierarchy is unavailable");
    auto directory=(boundary/relative).lexically_normal();
    if(directory.has_relative_path()&&directory.filename().empty())directory=directory.parent_path();
    std::error_code status;const auto exists=fs::is_directory(directory,status);
    require_memory_admission(!status&&exists,"Native host cgroup memory hierarchy is unavailable");
    for(;;directory=directory.parent_path()) {
      const auto limit=directory/(unified?"memory.max":"memory.limit_in_bytes");
      const auto limited=fs::exists(limit,status);
      require_memory_admission(!status,"Native host cgroup limit is unavailable");
      if(limited) {
        std::ifstream limits(limit);std::string value;limits>>value;const auto maximum=host_memory_number(value);
        if(maximum) {
          std::ifstream usage(directory/(unified?"memory.current":"memory.usage_in_bytes"));usage>>value;
          require_memory_admission(bool(usage),"Native host cgroup usage is unavailable");
          const auto current=host_memory_number(value);
          require_memory_admission(current.has_value(),"Native host cgroup usage is invalid");
          *available=std::min(*available,*current>=*maximum?size_t(0):*maximum-*current);
        }
      }
      if(directory==boundary)return *available;
    }
  }
  throw MemoryAdmissionError("Native host cgroup memory mount is unavailable");
}
}
