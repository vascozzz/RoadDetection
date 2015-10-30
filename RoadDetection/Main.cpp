#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"
#include "RoadDetection.h"

using namespace std;
using namespace cv;

String baseAssetsPath = "../Assets/";

void displayHelp();
void processImage(String path);
void processVideo(String path);

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		displayHelp();
		return -1;
	}

	int method = atoi(argv[1]);
	String file = (string)argv[2];

	switch (method)
	{
		case 1:
			processImage(baseAssetsPath + file);
			break;
		case 2:
			processVideo(baseAssetsPath + file);
			break;
		default:
			displayHelp();
			return -1;
	}

	waitKey(0);
}

void processImage(String path)
{
	RoadDetection roadDetector = RoadDetection(path);
	Mat result = roadDetector.processImage();

	namedWindow("Road Detection");
	imshow("Road Detection", result);
}

void processVideo(String path)
{
	VideoCapture cap = VideoCapture(path);
	RoadDetection roadDetector;

	namedWindow("Road Detection");

	if (!cap.isOpened())
	{
		cout << "Unable to read from file. Now exiting..." << endl;
		exit(1);
	}

	while (waitKey(1) < 0)
	{
		Mat rawFrame;
		
		if (!cap.read(rawFrame))
		{ 
			cout << "Skipped a frame..." << endl;
		}

		rawFrame = roadDetector.processVideo(rawFrame);
		
		imshow("Road Detection", rawFrame);
	}
}

void displayHelp()
{
	cout << "Usage example: RoadDetection.exe -method -filepath" << endl;
	cout << "Available methods: image(1), video(2)" << endl;
}
