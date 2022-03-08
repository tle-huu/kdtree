add_executable(find_nn main.cpp)
target_link_libraries(find_nn PRIVATE kdtree)
setWarnings(find_nn)
setStandard(find_nn)

# Profiling
target_compile_options(find_nn PRIVATE -O1)
target_link_libraries(find_nn PRIVATE) 

# TODO: remove this
configure_file(draw.py draw.py COPYONLY)
