SET(${PROJECT_NAME}_PYTHON_FILES
  __init__.py
  contact_fwddyn_aslr.py
  floating_actuation.py
  residual_frame_placement.py
  solver.py
  statemultiasr.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(aslr_to ${python})
  INSTALL(FILES aslr_to/${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
