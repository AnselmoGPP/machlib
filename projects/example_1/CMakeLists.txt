CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

if( UNIX )
	#set(CMAKE_CXX_COMPILER "/usr/bin/c++")
	#set(CMAKE_C_COMPILER "gcc-9")
	#set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
endif()

PROJECT(example_1
	VERSION 1.0
	DESCRIPTION "Testing"
	LANGUAGES CXX
	)

set(appConfig "Debug")
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_GENERATOR_PLATFORM x64)

MESSAGE(STATUS "Project: " ${PROJECT_NAME})

ADD_DEFINITIONS(
	-std=c++17
	#-O2		# Optimization incompatible with Debug mode ('/O2' and '/RTC1' are incompatible)
	#-O3
)
#ADD_COMPILE_DEFINITIONS( IMGUI_IMPL_OPENGL_LOADER_GLEW=1 )
#ADD_COMPILE_DEFINITIONS( IMGUI_IMPL_OPENGL_LOADER_GLAD=1 )

ADD_EXECUTABLE( ${PROJECT_NAME}
	src/main.cpp
)

TARGET_SOURCES(${PROJECT_NAME} PRIVATE
	#../machlib/src/machlib.cpp
)

TARGET_INCLUDE_DIRECTORIES( ${PROJECT_NAME} PUBLIC
	include
	../machlib/include
	../../extern/eigen
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} 
	../../_BUILD/machlib/Debug/machlib
)
