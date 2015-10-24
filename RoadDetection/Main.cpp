#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"
#include "RoadDetection.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "Usage example: RoadDetection.exe -method -filepath" << endl;
		cout << "Available methods: img, vid" << endl;
		return -1;
	}

	RoadDetection roadDetector;
	String method = (string)argv[1];
	String filePath = (string)argv[2];

	// image processing
	if (method == "img")
	{
		roadDetector = RoadDetection("../Assets/" + filePath);
		roadDetector.detectAll();
		roadDetector.displayControls();
	}

	// video processing
	else if (method == "vid")
	{
		VideoCapture cap = VideoCapture("../Assets/" + filePath);

		if (!cap.isOpened())
		{
			cout << "Unable to read from file. Now exiting..." << endl;
			return -1;
		}

		while (waitKey(10) < 0)
		{
			Mat frame;
			cap >> frame;

			roadDetector = RoadDetection(frame);
			roadDetector.detectAll();
		}
	}

	waitKey(0);
}
