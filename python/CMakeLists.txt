SET(${PROJECT_NAME}_PYTHON_FILES
  __init__.py
  free_fwddyn_aslr.py
  contact_fwddyn_aslr.py
  statemultibody_aslr.py
  statefloating_aslr.py
  actuation_aslr.py
  floating_actuation.py
  residual_frame_placement.py
  solver.py
  integrated_action.py
  contact3d_asr.py
  statemultiasr.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(aslr_to ${python})
  INSTALL(FILES aslr_to/${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
