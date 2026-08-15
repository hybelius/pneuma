add_library(fft)

target_sources(fft
  PRIVATE
    fft.cpp

  PUBLIC
    FILE_SET HEADERS
    FILES
      fft.h
)

target_include_directories(fft PUBLIC "${PROJECT_SOURCE_DIR}/src")
