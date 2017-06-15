#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

int main(int argc, char** argv)
{
	// Check to make sure the input file has been specified as an argument
	if(argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <video_file> " << std::endl;
		return -1;
	}

	// Create a VideoCapture object that handles opening the video file
	const std::string filename(argv[1]);
	cv::VideoCapture capture(filename);

	// Verify that the VideoCapture object has correctly opened the file
	if (!capture.isOpened())
	{
		std::cout << " Error. Failed to open camera/video file. " << std::endl;
		return -1;
	}

	// Create an output window for showing the image
	cv::namedWindow("Video Output", cv::WINDOW_AUTOSIZE);

  //First I capture only the 0th frame
  cv::Mat frame0,gray0;
  capture.grab();
  capture.retrieve(frame0,0);
  if( !frame0.data )
  { 
    std::cout<< " Error reading the image " << std::endl; 
  }
  cv::cvtColor(frame0,gray0,CV_RGB2GRAY);
  cv::imshow("0th Frame",gray0); 
  //0th frame capture complete
   
  //Step1: Detection of Keypoints using SURF detector for 1st video frame
  int minHessian = 100;
  cv::SurfFeatureDetector detector( minHessian );
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  detector.detect( gray0, keypoints_1 );

  //Step2: Calculating Descriptors for keypoints of the 1st video frame
  cv::SurfDescriptorExtractor extractor;
  cv::Mat descriptors_1, descriptors_2;
  extractor.compute( gray0, keypoints_1, descriptors_1 );
   
	while(true)
	{
		cv::Mat frame;

		// This method is recommended if you're using multiple cameras or videos simultaneously
		capture.grab();
		if(!capture.retrieve(frame,0))
		{
		 std::cout << "No image grabbed. Terminating capture." << std::endl;
		break;
		}

	
    cv::Mat im_gray;
    cv::cvtColor(frame,im_gray,CV_RGB2GRAY);
		cv::imshow("Video Output",im_gray); 
    detector.detect( im_gray, keypoints_2);//keypoint detection
    extractor.compute(im_gray, keypoints_2, descriptors_2 );//computation of descriptors

    //Step 3: Matching descriptor vectors using FLANN matcher 
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
        
    double max_dist = 0; double min_dist = 100;//Initialization

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < matches.size(); i++ )
    { 
      double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \t", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
  
   std::vector< cv::DMatch > good_matches;

    for( int i = 0; i <matches.size(); i++ )
    { 
      if( matches[i].distance <= cv::max(3*min_dist, 0.02) )
        {
        	good_matches.push_back( matches[i]); 
        }
    }

    //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches( frame0, keypoints_1, frame, keypoints_2,
                good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    //imshow( "Good Matches", img_matches );
       

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
      {
       //-- Get the keypoints from the good matches
       obj.push_back(keypoints_1[good_matches[i].queryIdx].pt );
       scene.push_back(keypoints_2[good_matches[i].trainIdx].pt );
      }

    std::cout<<"Good Matches : " << good_matches.size()<<std::endl;

    //Step 4: Matching descriptor vectors using FLANN matcher 
    cv::Mat H = findHomography(obj, scene,CV_RANSAC);

   

    std::vector<cv::Point2f> scene_corners(4);
    std::vector<cv::Point2f> obj_corners(4);
    
    //Storing the corners of rectangle containing the bulb in the first frame, this can be done before(outside) the loop
    obj_corners[0]=cv::Point2f(24,44);
    obj_corners[1]=cv::Point2f(196,44);
    obj_corners[2]=cv::Point2f(196,210);
    obj_corners[3]=cv::Point2f(24,210);

    //Step 4: Applying the perscpectiveTransform to obj_corners(rect corners for 1st frame) to obtain scene_corners
    //(rect corners for ith frame)
    cv::perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv::line( img_matches, scene_corners[0] + cv::Point2f( gray0.cols, 0), scene_corners[1] + cv::Point2f( gray0.cols, 0), cv::Scalar(0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[1] + cv::Point2f( gray0.cols, 0), scene_corners[2] + cv::Point2f( gray0.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[2] + cv::Point2f( gray0.cols, 0), scene_corners[3] + cv::Point2f( gray0.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[3] + cv::Point2f( gray0.cols, 0), scene_corners[0] + cv::Point2f( gray0.cols, 0), cv::Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches);

		// Wait for 33 milliseconds before processing the next frame of the video
		// If the user presses a key while waiting, then terminate the program.
		if(cv::waitKey(33) >= 0) break;
	}

	return 0;
}