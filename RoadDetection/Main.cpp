#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"
#include "RoadDetection.h"

using namespace std;
using namespace cv;

RoadDetection roadDetector;

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cout << "Usage example RoadDetection.exe -method --imagepath." << endl;
		return -1;
	}

	if ((string)argv[1] == "vid")
	{
		VideoCapture  cap = VideoCapture("../Assets/" + (string)argv[2]);

		while (waitKey(10) < 0)
		{
			Mat frame;
			cap >> frame;
			
			roadDetector = RoadDetection(frame);
			roadDetector.detectAll();
		}
	}

	else if ((string)argv[1] == "cam")
	{

	}

	else if ((string)argv[1] == "img")
	{
		Mat frame;
		roadDetector = RoadDetection("../Assets/" + (string)argv[2]);
		roadDetector.detectAll();
	}

	waitKey(0);
}
