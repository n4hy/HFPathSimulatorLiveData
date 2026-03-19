#!/bin/bash
# HF Path Simulator health check script

set -e

# Check if API is responding
curl -sf http://localhost:${HFPATHSIM_PORT:-8000}/api/v1/health > /dev/null

exit 0
