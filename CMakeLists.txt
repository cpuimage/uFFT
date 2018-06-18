cmake_minimum_required(VERSION 3.9)
project(uFFT_demo)

add_executable(uFFT_demo example.c uFFT.c)
target_link_libraries(uFFT_demo -lm)