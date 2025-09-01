# lives in cpp/
find_package(Akantu REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Boost REQUIRED COMPONENTS graph)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SRCS
  ${SRC_DIR}/main.cc
  ${SRC_DIR}/helpers.cc
)

include_directories(${SRC_DIR})

add_akantu_simulation(cohesive_sim ${SRCS})

if (TARGET Boost::graph)
  target_link_libraries(cohesive_sim PRIVATE Boost::graph)
else()
  target_include_directories(cohesive_sim PRIVATE ${Boost_INCLUDE_DIRS})
  # Use the specific graph lib if available; otherwise all Boost libs
  if (DEFINED Boost_GRAPH_LIBRARY)
    target_link_libraries(cohesive_sim PRIVATE ${Boost_GRAPH_LIBRARY})
  else()
    target_link_libraries(cohesive_sim PRIVATE ${Boost_LIBRARIES})
  endif()
endif()
target_link_libraries(cohesive_sim PRIVATE HDF5::HDF5)
set_target_properties(cohesive_sim PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
