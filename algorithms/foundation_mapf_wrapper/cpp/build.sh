#!/bin/bash
# Build script for mapf_features_cpp extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building mapf_features_cpp extension..."
python setup.py build_ext --inplace

# Copy the .so file to parent directory for easy import
cp mapf_features_cpp*.so ../

echo "Build complete!"
echo "Extension installed at: $(ls ../mapf_features_cpp*.so)"
