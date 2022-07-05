SET(${PROJECT_NAME}_PYTHON_FILES
  __init__.py
  free_fwddyn_asr.py
  free_fwddyn_vsa.py
  statemultibody_aslr.py
  actuation_aslr.py
  residual_frame_placement.py
  integrated_action.py
  stiffness_cost.py
  actuation_vsa.py
  residual_acrobot.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(aslr_to ${python})
  INSTALL(FILES aslr_to/${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
