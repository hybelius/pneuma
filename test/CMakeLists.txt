
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.15.3
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(tests number_theory/primes.cpp transforms/trig.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain fft)
target_include_directories(tests PUBLIC "${PROJECT_SOURCE_DIR}/src")
catch_discover_tests(tests)
