message(STATUS "Configuring target: fastfft::fastfft_tests")


add_executable(fastfft_tests ${TEST_SOURCES})
add_executable(fastfft::fastfft_tests ALIAS fastfft_tests)

target_link_libraries(fastfft_tests
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        )

target_precompile_headers(fastfft_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/fastfft/common/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/fastfft/common/Exception.h
        ${PROJECT_SOURCE_DIR}/src/fastfft/common/Logger.h
        ${PROJECT_SOURCE_DIR}/src/fastfft/common/Types.h
        )

target_include_directories(fastfft_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests)

install(TARGETS fastfft_tests
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )

message("")
