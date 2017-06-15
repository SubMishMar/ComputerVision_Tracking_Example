// 1_LoadImage.cpp - Loads an image file and displays it on the screen

#include <iostream>
#include <opencv2/core/core.hpp>  //-- Contains structures and classes for holding and manipulating images
#include <opencv2/highgui/highgui.hpp> //-- Contains functions for displaying images on screen

// using namespace cv;
// using namespace std;

int main(int argc, char** argv)
{
	//-- Make sure the user has entered the location of the image as an argument
	if(argc != 2)
	{
		std::cout << " Usage: " << argv[0] << " <iamge_file> " <<  std::endl;
		return -1;
	}

	//-- Create the image object and load the image from the file
	cv::Mat image;
	image = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);

	//-- Verify that the image was correctly loaded
	if(!image.data)
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//-- Create a window, and then display the image on this window
	cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display Window", image);

	//-- Wait for the user to press a key
	cv::waitKey(0);

	//-- End of the program. All images are automatically unloaded and deleted
	//--  from memory and all windows are destroyed when they go out of scope
	return 0;
}