set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++11 -pthread -pipe")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")
 
# include FindBANG.cmake and check cncc
find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
elseif (NOT BANG_CNCC_EXECUTABLE)
  message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should che
ck path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
endif()
 
# setup cncc flags
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -Wall -Werror -std=c++11 -pthread")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_220"
                                         "--bang-mlu-arch=mtp_270"
                                         "--bang-mlu-arch=mtp_290"
                                         "--bang-mlu-arch=mtp_372"
)
message("--------------------------------------------------")
message("CMAKE_SOURCE_DIR:${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE src_files ${src_files} "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/cpu/mluop/*.mlu")
bang_add_library(mluop SHARED ${src_files})
target_link_libraries(mluop cnrt cndrv)

message("----------------mlu build end--------------------")