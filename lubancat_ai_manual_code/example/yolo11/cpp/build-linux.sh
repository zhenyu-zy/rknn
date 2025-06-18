#!/bin/bash
set -e

echo "$0 $@"
while getopts ":t:b:dmr" opt; do
  case $opt in
    t)
      TARGET_SOC=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    m)
      ENABLE_ASAN=ON
      export ENABLE_ASAN=TRUE
      ;;
    r)
      DISABLE_RGA=ON
      ;;
    d)
      ENABLE_DMA32=ON
      export ENABLE_DMA32=TRUE
      ;;
    :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

if [ -z ${TARGET_SOC} ] ; then
  echo "$0 -t <target> "
  echo ""
  echo "    -t : target (rk356x/rk3588/rk3576)"
  echo "    -r : disable rga, use cpu resize image"
  echo "    -b : build_type(Debug/Release)"
  echo "    -m : enable address sanitizer, build_type need set to Debug"
  echo "such as: $0 -t rk3588 "
  echo "such as: $0 -t rk3588 -b Debug -m"
  echo ""
  exit -1
fi

case ${TARGET_SOC} in
    rk356x)
        ;;
    rk3588)
        ;;
    rk3576)
        TARGET_SOC="rk3576"
        ;;
    rk3566)
        TARGET_SOC="rk356x"
        ;;
    rk3568)
        TARGET_SOC="rk356x"
        ;;
    rk3562)
        TARGET_SOC="rk356x"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3562,rk3566,rk3568,rk3576,rk3588"
        exit -1
        ;;
esac

#RGA2 only support under 4G memory
if [[ -z ${ENABLE_DMA32} ]];then
    ENABLE_DMA32=OFF
fi

if [[ -z ${DISABLE_RGA} ]];then
    DISABLE_RGA=OFF
fi

# Debug / Release
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

# Build with Address Sanitizer for memory check, BUILD_TYPE need set to Debug
if [[ -z ${ENABLE_ASAN} ]];then
    ENABLE_ASAN=OFF
fi

# 板卡上编译，默认编译器
GCC_COMPILER=aarch64-linux-gnu
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_SOC}_linux
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SOC}_linux

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "ENABLE_DMA32=${ENABLE_DMA32}"
echo "DISABLE_RGA=${DISABLE_RGA}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DENABLE_DMA32=${ENABLE_DMA32} \
    -DENABLE_ASAN=${ENABLE_ASAN} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DDISABLE_RGA=${DISABLE_RGA} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install