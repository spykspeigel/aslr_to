CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(PROJECT_ORG spykspeigel)
SET(PROJECT_NAME aslr_to)
SET(PROJECT_DESCRIPTION "Prototype code for implementing optimal control problem for soft actuated robot in Crocoddyl.")
SET(PROJECT_URL https://github.com/${PROJECT_ORG}/${PROJECT_NAME})

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/test.cmake)
INCLUDE(cmake/python.cmake)

IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
ENDIF()

SETUP_PROJECT()

# Project configuration
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find-external/CppAD/" ${CMAKE_MODULE_PATH})

FINDPYTHON()
ADD_PROJECT_DEPENDENCY(crocoddyl)
ADD_PROJECT_DEPENDENCY(pinocchio)
ADD_PROJECT_DEPENDENCY(example-robot-data 3.8.0 REQUIRED PKG_CONFIG_REQUIRES "example-robot-data >= 3.8.0")

ADD_SUBDIRECTORY(python)
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(unittest)
ENDIF(BUILD_TESTING)
#ADD_SUBDIRECTORY(examples)
