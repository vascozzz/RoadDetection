#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include "DetectionTimer.h"
#include "Line.h"

using namespace std;
using namespace cv;

class RoadDetection
{
private:
	Mat original;

	int blurKernel = 5;
	int cannyLowThresh = 50, cannyHighThresh = 200;
	int houghThresh = 200, houghMinLines = 5;
	int houghProbThresh = 120, houghProbMinLineLength = 80, houghProbMaxLineGap = 50;
	int cascadeMinNeighbors = 2, cascadeMinSize = 30, cascadeMaxSize = 30;

	float houghLowAngle = 1.35f, houghHighAngle = 1.75f;
	float houghProbLowAngle = 0.25f, houghProbHighAngle = 5.85f;
	double cascadeScale = 1.05f;
	double maxLineAngleDiff = 0.0872665;
	int maxLineDistDiff = 20;

	double getAngleBetweenPoints(Point pt1, Point pt2);
	double getDistBetweenPoints(Point pt1, Point pt2);
	Point getPointAverage(Point pt1, Point pt2);
	Point getLineIntersection(Line l1, Line l2);
	Point getVanishingPoint(vector<Line> lines, Size frameSize);

	vector<Line> getHoughLines(Mat frame);
	vector<Line> getMainLines(vector<Line> lines);
	vector<Line> getHoughProbLines(Mat frame);
	vector<Line> getFilteredLines(vector<Line> lines);

	vector<Rect> getVehicles(Mat frame);

public:
	RoadDetection(){};
	RoadDetection(String filePath);
	RoadDetection(Mat original);
	~RoadDetection(){};

	void setFile(String path);
	void setFile(Mat original);

	void method1();
	void method2();
	void method3();

	void detectAll();
	void displayControls();
};