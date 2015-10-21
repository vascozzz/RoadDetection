#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"

using namespace std;
using namespace cv;

class RoadDetection
{
private:
	Mat original;
	double getAngleBetweenPoints(Point pt1, Point pt2);
public:
	RoadDetection(){};
	RoadDetection(String filePath);
	RoadDetection(Mat original);
	~RoadDetection(){};

	void setFile(String path);
	void setFile(Mat original);
	void method1();
	void method2();

	int imgExample();
	int videoExample();
};