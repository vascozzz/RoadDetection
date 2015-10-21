#include "RoadDetection.h"

RoadDetection::RoadDetection(String filePath)
{
	original = imread(filePath);
}

RoadDetection::RoadDetection(Mat original)
{
	original.copyTo(this->original);
}

void RoadDetection::setFile(String filePath)
{
	original = imread(filePath);
}

void RoadDetection::setFile(Mat original)
{
	original.copyTo(this->original);
}

double RoadDetection::getAngleBetweenPoints(Point pt1, Point pt2)
{
	double xDiff = pt2.x - pt1.x;
	double yDiff = pt2.y - pt1.y;

	if (atan2(yDiff, xDiff) < 0)
	{
		return atan2(yDiff, xDiff) + 2 * CV_PI;
	}
	else
	{
		return atan2(yDiff, xDiff);
	}
}

void RoadDetection::method1()
{
	Mat originalImg, grayImg, erodedImg, dilatedImg, morphImg, pathImg;
	DetectionTimer timer;

	// save a copy of the original image
	original.copyTo(originalImg);

	// convert to grayscale, threshold (if pixel > thresh then pixel = max, otherwise 0)
	cvtColor(originalImg, grayImg, CV_BGR2GRAY);
	threshold(grayImg, grayImg, 100, 255, THRESH_BINARY);

	// initiate timer, image transformations
	timer.start();

	// eroding process (picks local minimum pixel value from kernel; increases dark areas)
	erode(grayImg, erodedImg, Mat(), Point(2, 2), 7);

	// dilating process (picks local maximum pixel value from kernel; increases light areas)
	dilate(grayImg, dilatedImg, Mat(), Point(2, 2), 7);

	// threshold dilated image (if pixel > thresh then pixel = 0, otherwise max)
	threshold(dilatedImg, dilatedImg, 1, 50, THRESH_BINARY_INV);

	// add both (resulting image has the lowest pixel value between both images; creates contrasting zones)
	morphImg = Mat(grayImg.size(), CV_8U, Scalar(0));
	morphImg = erodedImg + dilatedImg;

	// watershed (converting from unsigned 8-bit per pixel (0-255) to signed 32-bit for math operations, and then back for display)
	morphImg.convertTo(pathImg, CV_32S);
	watershed(originalImg, pathImg);
	pathImg.convertTo(pathImg, CV_8U);

	// display
	namedWindow("input");
	namedWindow("path");

	imshow("input", originalImg);
	imshow("path", pathImg);

	// process time
	cout << "time: " << timer.getElapsed() << endl;
}

void RoadDetection::method2()
{
	Mat originalImg, testImg, blurredImg, contoursImg, contoursInvImg, houghImg, houghPImg, pathImg;
	vector<Vec2f> houghLines;
	vector<Vec4i> houghPLines;
	int houghVote = 200;

	// save a copy of the original image
	original.copyTo(originalImg);

	// attempt noise removal
	GaussianBlur(originalImg, blurredImg, Size(3, 3), 0, 0);

	// canny edge detection (should check thresh values), inverted
	Canny(blurredImg, contoursImg, 50, 350);
	threshold(contoursImg, contoursInvImg, 128, 255, THRESH_BINARY_INV);

	// hough transform, with voting process ensuring at least 5 different lines
	while (houghLines.size() < 5 && houghVote > 0){
		HoughLines(contoursImg, houghLines, 1, CV_PI / 180, houghVote, 0, 0);
		houghVote -= 5;
	}

	// display hough transform lines
	houghImg = Mat(originalImg.size(), CV_8U, Scalar(0));

	for (size_t i = 0; i < houghLines.size(); i++)
	{
		float rho = houghLines[i][0];
		float theta = houghLines[i][1];

		// filter out horizontal lines by comparing theta (angle of lines perpendicular to the detected line)
		if (theta < 1.25f || theta > 1.85f)
		{
			Point pt1, pt2;
			double a = cos(theta);
			double b = sin(theta);
			double x0 = a*rho;
			double y0 = b*rho;

			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));

			line(houghImg, pt1, pt2, Scalar(255, 0, 0), 3, CV_AA);
		}
	}

	// probabilistic hough
	houghPImg = Mat(originalImg.size(), CV_8U, Scalar(0));
	HoughLinesP(contoursImg, houghPLines, 1, CV_PI / 180, 30, 60, 10);

	for (size_t i = 0; i < houghPLines.size(); i++)
	{
		Vec4i houghLine = houghPLines[i];
		Point pt1 = Point(houghLine[0], houghLine[1]);
		Point pt2 = Point(houghLine[2], houghLine[3]);
		double angle = getAngleBetweenPoints(pt1, pt2);
		
		// filter out horizontal lines by comparing angle between points
		if (angle > 0.1 && angle < 6)
		{
			line(houghPImg, pt1, pt2, Scalar(255, 0, 0), 3, CV_AA);
		}
	}

	// bitwise AND of the two hough transforms
	pathImg = Mat(originalImg.size(), CV_8U, Scalar(0));
	bitwise_and(houghImg, houghPImg, pathImg);

	// display
	namedWindow("input");
	namedWindow("path");

	imshow("input", originalImg);
	imshow("path", pathImg);
}

int RoadDetection::imgExample()
{
	Mat img1, img2, img3;
	String inputImage = "C:\\Programming\\C++\\OpenCV_Testbed\\OpenCV_Testbed\\logo.png";

	img1 = imread(inputImage, IMREAD_COLOR);

	if (img1.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	img2 = img1;
	img1.copyTo(img3);
	flip(img2, img2, 1);

	namedWindow("img1", WINDOW_AUTOSIZE);
	namedWindow("img2", WINDOW_AUTOSIZE);
	namedWindow("img3", WINDOW_AUTOSIZE);

	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);

	waitKey(0);
	return 0;
}

int RoadDetection::videoExample()
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		return -1;
	}

	Mat edges;
	namedWindow("original", 1);
	namedWindow("edges", 1);

	for (;;)
	{
		Mat frame;
		cap >> frame;
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);

		imshow("edges", edges);
		imshow("original", frame);

		if (waitKey(30) >= 0) {
			break;
		}
	}
	return 0;
}