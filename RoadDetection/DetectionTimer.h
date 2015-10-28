#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class DetectionTimer
{
private:
	double initialTime;

public:
	DetectionTimer();
	~DetectionTimer(){};

	void start();
	double getElapsed();
};
