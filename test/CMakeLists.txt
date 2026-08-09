
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.15.3
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_primes number_theory/primes.cpp)
target_link_libraries(test_primes PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_primes PUBLIC "${PROJECT_SOURCE_DIR}/src")
