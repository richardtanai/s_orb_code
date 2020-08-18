cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()



add_definitions(-DCOMPILEDWITHC11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w   -O3 -march=native -fopenmp")

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE("CMAKE Version: " ${CMAKE_VERSION})



LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(realsense2 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

MESSAGE("Pangolin Dirs: " ${Pangolin_INCLUDE_DIRS})
MESSAGE("Eigen Dirs: " ${EIGEN3_INCLUDE_DIR})

include_directories("/home/richard/dependencies/tensorflow") #change path
include_directories("/home/richard/dependencies/tensorflow/bazel-genfiles") #change path
include_directories("/home/richard/dependencies/tensorflow/tensorflow") #change path
include_directories("/home/richard/dependencies/tensorflow/third_party") #change path



### BUILD FULL ###


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)



add_library(${PROJECT_NAME} SHARED
src/RunSemSeg.cc
src/pointcloudmapping.cc
src/SemSeg
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so 
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so 
/home/richard/dependencies/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so #change path
${PCL_LIBRARIES}
)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D/bin)



add_executable(rgbd_tum_seg Examples/RGB-D/rgbd_tum_seg.cc)
target_link_libraries(rgbd_tum_seg ${PROJECT_NAME})

add_executable(rgbd_tum_seg_step Examples/RGB-D/rgbd_tum_seg_step.cc)
target_link_libraries(rgbd_tum_seg_step ${PROJECT_NAME})

add_executable(rgbd_tum_seg_no_loop Examples/RGB-D/rgbd_tum_seg_no_loop.cc)
target_link_libraries(rgbd_tum_seg_no_loop ${PROJECT_NAME})



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/realsense/bin)

add_executable(rs_live realsense/rs_live.cc)
target_link_libraries(rs_live ${PROJECT_NAME} ${realsense2_LIBRARY})

add_executable(rs_play realsense/rs_playback.cc)
target_link_libraries(rs_play ${PROJECT_NAME} ${realsense2_LIBRARY})

add_executable(rs_fbf realsense/rs_frame_by_frame.cc)
target_link_libraries(rs_fbf ${PROJECT_NAME} ${realsense2_LIBRARY})

add_executable(rs_fbf_step realsense/rs_frame_by_frame_step.cc)
target_link_libraries(rs_fbf_step ${PROJECT_NAME} ${realsense2_LIBRARY})

add_executable(get_ini realsense/get_ini.cc)
target_link_libraries(get_ini ${PROJECT_NAME} ${realsense2_LIBRARY})

