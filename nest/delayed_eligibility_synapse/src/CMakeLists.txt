set(MODULE_SOURCES
    mymodule.cpp
    stdp_delayed_eligibility_synapse.h
    stdp_delayed_eligibility_synapse.cpp
)

add_library( ${MODULE_NAME}_module MODULE ${MODULE_SOURCES} )
target_link_libraries(${MODULE_NAME}_module ${USER_LINK_LIBRARIES})
set_target_properties( ${MODULE_NAME}_module
   PROPERTIES
   COMPILE_FLAGS "${NEST_CXXFLAGS} -DLTX_MODULE"
   LINK_FLAGS "${NEST_LIBS}"
   PREFIX ""
   OUTPUT_NAME ${MODULE_NAME} )
install( TARGETS ${MODULE_NAME}_module
   DESTINATION ${CMAKE_INSTALL_LIBDIR}/nest
   )
