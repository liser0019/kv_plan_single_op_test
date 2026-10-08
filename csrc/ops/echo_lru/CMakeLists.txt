set(_echo_kernel "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/echo_lru_launch.cpp")
register_direct_launch_op(
    "${_echo_kernel}"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/echo_lru_plugin.cpp"
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
set_property(GLOBAL APPEND PROPERTY DIRECT_LAUNCH_PURE_SIMT_SRCS "${_echo_kernel}")
