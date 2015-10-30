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
	Point previousVanishingPoint;
	vector<Line> previousLines;

	int BLUR_KERNEL = 5;
	int CANNY_MIN_THRESH = 50, CANNY_MAX_THRESH = 200;
	int HOUGH_DEFAULT_THRESH = 175, HOUGH_MIN_THRESH = 200, HOUGH_MIN_LINES = 5;
	int HOUGH_PROB_THRESH = 120, HOUGH_PROB_MIN_LINE_LENGTH = 80, HOUGH_PROB_MAX_LINE_GAP = 50;
	int CASCADE_MIN_NEIGHBORS = 2, CASCADE_MIN_SIZE = 30, CASCADE_MAX_SIZE = 30;
	int LINE_DIST_MAX_DIFF = 20;

	float HOUGH_MIN_ANGLE = 1.35f, HOUGH_MAX_ANGLE = 1.75f;
	float HOUGH_PROB_MIN_ANGLE = 0.25f, HOUGH_PROB_MAX_ANGLE = 5.85f;
	double CASCADE_SCALE = 1.05f;
	double LINE_DIFF_MAX_ANGLE = 0.0872665;
	String CLASSIFIER_PATH = "../Assets/cars.xml";

	double getDistBetweenPoints(Point pt1, Point pt2);
	Point getPointAverage(Point pt1, Point pt2);
	Point getLineIntersection(Line l1, Line l2);
	Point getVanishingPoint(vector<Line> lines, Size frameSize);

	vector<Line> getHoughLines(Mat frame, bool useMinimum);
	vector<Line> getMainLines(vector<Line> lines);
	vector<Line> getHoughProbLines(Mat frame);
	vector<Line> getFilteredLines(vector<Line> lines);
	vector<Line> getLimitedLines(vector<Line> lines, int offset);
	vector<Line> shiftLines(vector<Line> lines, int shift);
	vector<Point> getRoadShape(Mat screen, Line l1, Line l2, Point inter);
	vector<Rect> getVehicles(Mat frame);

	void drawLines(Mat frame, vector<Line> lines, Scalar color, int thickness, int offset);
	void drawCircle(Mat frame, Point center, Scalar color, int radius, int thickness, int offset);
	void drawRects(Mat frame, vector<Rect> rects, Scalar color, int thickness, int offset);
	void drawRoadShape(Mat frame, vector<Point> points, Scalar color, double alpha, int offset);
	void combineWithSection(Mat frame, Mat section, int initialPos, int offset);

public:
	RoadDetection(){};
	RoadDetection(String filePath);
	RoadDetection(Mat original);
	~RoadDetection(){};

	void setFile(String path);
	void setFile(Mat original);

	Mat processImage();
	Mat processVideo(Mat rawFrame);

	void displayControls();
};