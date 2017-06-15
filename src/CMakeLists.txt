# Always required at the beginning of a cmake file
# Checks to make sure that you have a correct version
cmake_minimum_required(VERSION 2.8)

# Always required at the beginning of a cmake file
# Creates a "Project" name that can be used with other cmake files
project( DisplayImage )

# Look for files of external libraries required by your project
find_package( OpenCV REQUIRED )

# The first argument is the name of the output file, and all others are the files to compile into this output
add_executable( LoadImage 1_LoadImage.cpp )
# You need to explicitly mention which external libraries you want to include into this output
target_link_libraries( LoadImage ${OpenCV_LIBS} )

# The first argument is the name of the output file, and all others are the files to compile into this output
add_executable( LoadVideo 2_LoadVideo.cpp )
# You need to explicitly mention which external libraries you want to include into this output
target_link_libraries( LoadVideo ${OpenCV_LIBS} )

add_executable( Load_Video 3_LoadVideo.cpp )
# You need to explicitly mention which external libraries you want to include into this output
target_link_libraries( Load_Video ${OpenCV_LIBS} )
