#!/bin/bash
# conda-forge build script for hfpathsim

set -ex

# Install the package
${PYTHON} -m pip install . -vv --no-deps --no-build-isolation

# Create py.typed marker
touch ${SP_DIR}/hfpathsim/py.typed
