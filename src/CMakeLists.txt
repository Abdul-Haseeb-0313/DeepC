set(DEEPC_SRC data_processing.c losses.c models.c layers.c matrix.c
optimizers.c)

add_library(deepc STATIC ${DEEPC_SRC})
target_include_directories(deepc PUBLIC ../include)
target_link_libraries(deepc PUBLIC m)
