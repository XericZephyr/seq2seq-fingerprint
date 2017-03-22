# !/bin/bash

# Return error if any script returns a non-zero exit code.
set -e

# perform lint test for all files
pylint -r n --output-format=colorized --extension-pkg-whitelist=numpy,tensorflow --disable=duplicate-code,no-member,fixme *.py unsupervised
