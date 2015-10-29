#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class Line
{
private:
	double getAngleBetweenPoints();
	float getSlope();
	float getIntercept();

public:
	Point pt1, pt2;
	double angle;
	float slope, intercept;

	Line(){};
	Line(Point pt1, Point pt2);
	~Line(){};
};

