#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DetectionTimer.h"

#define PI 3.1415926;

using namespace std;
using namespace cv;

class RoadDetection
{
private:
	Mat original;
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