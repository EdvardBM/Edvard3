cmake_minimum_required(VERSION 3.8)
project(edvard_rviz2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(edvard_interfaces REQUIRED)

install(
  DIRECTORY scripts config launch meshes rviz urdf 
  DESTINATION share/${PROJECT_NAME}
)

# Install joint_state_republisher.py script and other scripts
install(PROGRAMS
  scripts/joint_state_republisher.py
  scripts/joint_state_republisher_to_robot.py
  DESTINATION lib/${PROJECT_NAME}
)


# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Ament package generation
ament_package()
