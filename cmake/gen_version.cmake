# Script mode (cmake -P): stamp the build-time version header.
# Inputs: SRC_DIR, OUT_FILE, BASE_VERSION, OVERRIDE, LIBINT_VER, FLAGS_STR
# The header is rewritten only when its content changes (copy_if_different),
# so an unchanged tree does not trigger recompilation.

execute_process(COMMAND git -C "${SRC_DIR}" rev-parse --short=8 HEAD
                OUTPUT_VARIABLE _hash OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _rc ERROR_QUIET)
if(NOT _rc EQUAL 0)
  set(_hash "unknown")
  set(_state "unknown")
else()
  execute_process(COMMAND git -C "${SRC_DIR}" status --porcelain --untracked-files=no
                  OUTPUT_VARIABLE _st RESULT_VARIABLE _rc2 ERROR_QUIET)
  if(NOT _rc2 EQUAL 0)
    set(_state "unknown")
  elseif(_st STREQUAL "")
    set(_state "clean")
  else()
    set(_state "modified")
  endif()
endif()

if(OVERRIDE)
  set(_ver "${OVERRIDE}")
else()
  set(_ver "${BASE_VERSION}")
endif()
if(NOT LIBINT_VER)
  set(LIBINT_VER "unknown")
endif()

set(_tmp "${OUT_FILE}.tmp")
file(WRITE "${_tmp}" "#ifndef LIBRA_VERSION_H\n#define LIBRA_VERSION_H\n#define LIBRA_VERSION_STRING \"${_ver}\"\n#define LIBRA_GIT_HASH \"${_hash}\"\n#define LIBRA_TREE_STATE \"${_state}\"\n#define LIBRA_LIBINT_VERSION \"${LIBINT_VER}\"\n#define LIBRA_FLAGS \"${FLAGS_STR}\"\n#endif\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_tmp}" "${OUT_FILE}")
file(REMOVE "${_tmp}")
