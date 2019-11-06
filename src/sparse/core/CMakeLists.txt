file(GLOB_RECURSE WRAPPED_SOURCES "*.cpp")
pybind11_add_module(core ${WRAPPED_SOURCES})
set_target_properties(core PROPERTIES LINK_SEARCH_START_STATIC 1)
set_target_properties(core PROPERTIES LINK_SEARCH_END_STATIC 1)
