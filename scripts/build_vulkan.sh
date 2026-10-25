#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
headers="${VULKAN_HEADERS:-/home/abhiram/Documents/ChatGPT/Mistral-Timestral-Experiment/vendor/Vulkan-Headers/include}"
glslc -O native/scorer.comp -o native/scorer.spv
c++ -O3 -shared -fPIC -I "$headers" native/scorer.cpp -o native/libscorer.so -l:libvulkan.so.1
glslc -O native/linear.comp -o native/linear.spv
c++ -O3 -shared -fPIC -I "$headers" native/linear.cpp -o native/liblinear.so -l:libvulkan.so.1
