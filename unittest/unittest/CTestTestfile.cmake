# CMake generated Testfile for 
# Source directory: /home/sp/Documents/aslr_to/unittest
# Build directory: /home/sp/Documents/aslr_to/unittest/unittest
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(py-free_fwd_aslr "/home/sp/miniconda3/bin/python" "/home/sp/Documents/aslr_to/unittest/test_free_fwd_aslr.py")
set_tests_properties(py-free_fwd_aslr PROPERTIES  ENVIRONMENT "PYTHONPATH=/home/sp/Documents/aslr_to/unittest/python/:/opt/ros/noetic/lib/python3/dist-packages:/usr/local/lib/python3.8/site-packages:/opt/openrobots/lib/python3.8/site-packages:" _BACKTRACE_TRIPLES "/home/sp/Documents/aslr_to/cmake/test.cmake;104;ADD_TEST;/home/sp/Documents/aslr_to/unittest/CMakeLists.txt;6;ADD_PYTHON_UNIT_TEST;/home/sp/Documents/aslr_to/unittest/CMakeLists.txt;0;")
