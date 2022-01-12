find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)
# doxygen and sphinx-build must be available.
# The sphinx extension "breathe" should be installed as well.

# ---------------------------------------------------------------------------------------
# Doxygen
# ---------------------------------------------------------------------------------------
set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml/index.xml)

# Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Get and config the input file for doxygen
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Create the noa-doxygen target
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        DEPENDS ${DOXYGEN_INPUT_DIR}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM)
add_custom_target(noa-doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

# ---------------------------------------------------------------------------------------
# Sphinx
# ---------------------------------------------------------------------------------------
set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/sphinx/index.html)

# Create the noa-sphinx target
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
        COMMAND
        ${SPHINX_EXECUTABLE} -b html
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.Noa=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}

        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/index.rst ${DOXYGEN_INDEX_FILE}
        MAIN_DEPENDENCY ${SPHINX_SOURCE_DIR}/conf.py
        COMMENT "Generating documentation with Sphinx")
add_custom_target(noa-sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
install(DIRECTORY ${SPHINX_BUILD_DIR}
        DESTINATION ${CMAKE_INSTALL_DOCDIR})
