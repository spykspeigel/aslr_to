SET(${PROJECT_NAME}_PYTHON_FILES
  __init__.py
  free_fwddyn_asr.py
  free_fwddyn_vsa.py
  free_fwddyn_vsa_qb.py
  contact_fwddyn.py
  statemultibody_aslr.py
  actuation_aslr.py
  residual_frame_placement.py
  solver.py
  integrated_action.py
  stiffness_cost.py
  actuation_vsa.py
  soft_residual_model.py
  actuation_condensed.py
  vsa_dynamics_residual.py
  residual_acrobot.py
  actuation_vsa_qb.py
  actuation_fishing.py
  free_fwddyn_fishing_2.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(aslr_to ${python})
  INSTALL(FILES aslr_to/${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
