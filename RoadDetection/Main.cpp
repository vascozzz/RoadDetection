#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"
#include "RoadDetection.h"

using namespace std;
using namespace cv;

/*
* @param argv[1] - method
* @param argv[2] - image path
*/
int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "Usage example RoadDetection.exe -method -imagepath." << endl;
		return -1;
	}

	String filePath = "../Assets/" + (String)argv[2] + ".png";
	RoadDetection detector(filePath);

	detector.method3();
	waitKey(0);
}
