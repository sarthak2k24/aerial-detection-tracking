cmake_minimum_required(VERSION 3.16)
project(aerial_detection_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn videoio highgui)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pure geometry + tracking logic: no OpenCV dependency, fully unit-testable
add_library(tracking_core
  src/Geometry.cpp
  src/Tracker.cpp
)
target_include_directories(tracking_core PUBLIC src)

# Detector wraps OpenCV DNN
add_library(detector src/Detector.cpp)
target_include_directories(detector PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detector PUBLIC tracking_core ${OpenCV_LIBS})

foreach(t tracking_core detector)
  if(MSVC)
    target_compile_options(${t} PRIVATE /W4)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra)
  endif()
endforeach()

add_executable(detect_track src/main.cpp)
target_link_libraries(detect_track PRIVATE detector)

enable_testing()
add_executable(tracking_tests
  tests/test_geometry.cpp
  tests/test_tracker.cpp
)
target_link_libraries(tracking_tests PRIVATE tracking_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tracking_tests)
