#!/usr/bin/env bash
set -euo pipefail

readonly source_dir="${1:-/src}"
readonly build_dir="${2:-${source_dir}}"
readonly install_dir="${3:-/src/install-cuda}"

cd "${source_dir}"

if [[ ! -d src/libsvm-3.35 ]]; then
  tar -xzf src/libsvm-3.35.tar.gz -C src
fi

if [[ ! -d src/dlib-20.0 ]]; then
  tar -xjf src/dlib-20.0.tar.bz2 -C src
fi

if [[ ! -x configure || configure.ac -nt configure ]]; then
  autoreconf -fi
fi

mkdir -p "${build_dir}" "${install_dir}"
cd "${build_dir}"

if [[ ! -f config.status ]]; then
  "${source_dir}/configure" \
    --prefix="${install_dir}" \
    --without-doc \
    --without-perl \
    --without-python \
    --without-python2 \
    --without-swig \
    --without-svm \
    --without-kinfold \
    --without-forester \
    --without-rnalocmin \
    --without-rnaxplorer \
    --disable-lto
fi

make -C src -j"$(nproc)"
make -C src/ViennaRNA install
make -C src/bin install
make -C misc install
make -C src/ViennaRNA/gpu -f Makefile.cuda \
  TOP_SRCDIR="${source_dir}" \
  PREFIX="${install_dir}"
