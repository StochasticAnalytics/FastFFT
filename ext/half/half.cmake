include(FetchContent)
message(STATUS "Fetching header dependency: half")
FetchContent_Declare(half-ieee754
        GIT_REPOSITORY git@github.com:ffyr2w/half-ieee754.git
        GIT_TAG master
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        )

FetchContent_MakeAvailable(half-ieee754)
message("")
