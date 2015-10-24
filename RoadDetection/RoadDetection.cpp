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
	Mat originalImg, testImg, blurredImg, contoursImg, contoursInvImg, houghImg, houghPImg, pathImg, carsImg;
	vector<Vec2f> houghLines;
	vector<Vec4i> houghPLines;
	int houghVote = 200;

	// save a copy of the original image
	original.copyTo(originalImg);

	// attempt noise removal
	GaussianBlur(originalImg, blurredImg, Size(5, 5), 0, 0);

	// canny edge detection (should check thresh values), inverted
	Canny(blurredImg, contoursImg, 50, 200);
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
		if (theta < 1.45f || theta > 1.65f)
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
	HoughLinesP(contoursImg, houghPLines, 1, CV_PI / 180, 50, 50, 100);

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
	bitwise_and(houghImg, houghImg, pathImg);

	cvtColor(pathImg, pathImg, CV_GRAY2BGR);
	Mat finalImg = originalImg + pathImg;

	// display
	namedWindow("input");
	namedWindow("path");
	namedWindow("final");

	imshow("input", originalImg);
	imshow("path", houghImg);
	imshow("final", contoursImg);
}

void RoadDetection::method3()
{
	Mat src, gray, contours, grayhc;
	vector<Vec4i> lines;
	const double scale = 2.0;

	// save a copy of the src image
	original.copyTo(src);

	// create a grayscaled version of src image
	cvtColor(src, gray, CV_BGR2GRAY);

	// create a scaled grayscaled and high contrast image (used to detect cars)
	grayhc = Mat(Size(cvRound(src.cols / scale), cvRound(src.rows / scale)), CV_8U, 1);
	resize(gray, grayhc, grayhc.size(), 0, 0, InterpolationFlags::INTER_LINEAR);
	equalizeHist(grayhc, grayhc);

	// smoothen grayscaled image and create contours image
	GaussianBlur(gray, gray, Size(5, 5), 0);
	Canny(gray, contours, 50, 200);

	// Calculate lines with Hough Probabilistic
	HoughLinesP(contours, lines, 1, CV_PI / 180, 150, 250, 100);

	// Process the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		//p1x p1y p2x p2y
		// 0   1   2   3
		int dx = lines[i][2] - lines[i][0];
		int dy = lines[i][3] - lines[i][1];

		//get angle
		double angle = atan2(dy, dx) < 0 ? atan2(dy, dx) * 180 / CV_PI + 360 : atan2(dy, dx) * 180 / CV_PI;
			
		if (abs(angle) < 10 || abs(angle) > 350)
			continue;

		if (lines[i][1] > lines[i][3] + 50 || lines[i][1] < lines[i][3] - 50)
		{
			Point p1(lines[i][0], lines[i][1]);
			Point p2(lines[i][2], lines[i][3]);
			line(src, p1, p2, Scalar(0, 0, 255), 2, CV_AA);
		}
	}

	//display
	namedWindow("src");
	namedWindow("gray");
	namedWindow("equalized");
	namedWindow("canny");

	imshow("src", src);
	imshow("gray", gray);
	imshow("equalized", grayhc);
	imshow("canny", contours);
}

void RoadDetection::detectAll()
{
	Mat originalFrame, blurredFrame, contoursFrame, houghFrame, houghProbFrame;
	vector<Vec4i> houghLines, houghProbLines;
	vector<Rect> vehicles;
	Point vanishingPoint;

	// save a copy of the original frame
	original.copyTo(originalFrame);

	// ROI
	Mat tmpFrame, workingFrame;
	original.copyTo(tmpFrame);
	workingFrame = tmpFrame(CvRect(0, original.rows / 2, original.cols, original.rows / 2));

	// smooth and remove noise
	GaussianBlur(originalFrame, blurredFrame, Size(blurKernel, blurKernel), 0, 0);

	// edge detection (canny, inverted)
	Canny(blurredFrame, contoursFrame, cannyLowThresh, cannyHighThresh);
	threshold(contoursFrame, contoursFrame, 128, 255, THRESH_BINARY);

	// hough transform lines
	houghFrame = Mat(originalFrame.size(), CV_8U, Scalar(0));
	houghLines = getHoughLines(contoursFrame);

	for (size_t i = 0; i < houghLines.size(); i++)
	{
		Point pt1 = Point(houghLines[i][0], houghLines[i][1]);
		Point pt2 = Point(houghLines[i][2], houghLines[i][3]);
		line(originalFrame, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}

	// probabilistic hough transform
	houghProbFrame = Mat(originalFrame.size(), CV_8U, Scalar(0));
	houghProbLines = getHoughProbLines(contoursFrame);

	for (size_t i = 0; i < houghProbLines.size(); i++)
	{
		Point pt1 = Point(houghProbLines[i][0], houghProbLines[i][1]);
		Point pt2 = Point(houghProbLines[i][2], houghProbLines[i][3]);
		line(originalFrame, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
	}

	// vanishing point
	vanishingPoint = getVanishingPoint(houghLines, originalFrame.size());

	if (vanishingPoint.x > -1) 
	{
		ellipse(originalFrame, vanishingPoint, Size(15, 15), 0, 0, 360, Scalar(255, 0, 255), 2, CV_AA, 0);
	}

	// vehicles
	vehicles = getVehicles(originalFrame);

	for (size_t i = 0; i < vehicles.size(); i++)
	{
		Point center(vehicles[i].x + vehicles[i].width / 2, vehicles[i].y + vehicles[i].height / 2);
		ellipse(originalFrame, center, Size(vehicles[i].width / 2, vehicles[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 1, CV_AA, 0);
	}

	namedWindow("original");
	imshow("original", originalFrame);
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

vector<Vec4i> RoadDetection::getHoughLines(Mat frame)
{
	vector<Vec2f> lines;
	vector<Vec4i> filteredLines;

	// lower thresh in an attempt to get at least houghMinLines
	while ((int)lines.size() < houghMinLines && houghThresh > 0)
	{
		HoughLines(frame, lines, 1, CV_PI / 180, houghThresh, 0, 0);
		houghThresh -= 5;
	}

	// convert rho and theta to actual points
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];

		if (theta < houghLowAngle || theta > houghHighAngle)
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

			Vec4i line = Vec4i(pt1.x, pt1.y, pt2.x, pt2.y);
			filteredLines.push_back(line);
		}
	}

	return filteredLines;
}

vector<Vec4i> RoadDetection::getHoughProbLines(Mat frame)
{
	vector<Vec4i> lines;
	vector<Vec4i> filteredLines;

	HoughLinesP(frame, lines, 1, CV_PI / 180, houghProbThresh, houghProbMinLineLength, houghProbMaxLineGap);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Point pt1 = Point(lines[i][0], lines[i][1]);
		Point pt2 = Point(lines[i][2], lines[i][3]);

		double angle = getAngleBetweenPoints(pt1, pt2);

		if (angle > houghProbLowAngle && angle < houghProbHighAngle)
		{
			filteredLines.push_back(lines[i]);
		}
	}

	return filteredLines;
}

Point RoadDetection::getVanishingPoint(vector<Vec4i> lines, Size frameSize)
{
	vector<Point> interPoints;

	for (size_t i = 0; i < lines.size(); i++)
	{
		for (size_t j = i + 1; j < lines.size(); j++)
		{
			Vec4i l1 = lines[i];
			Vec4i l2 = lines[j];

			Point l1p1 = Point(l1[0], l1[1]);
			Point l1p2 = Point(l1[2], l1[3]);
			Point l2p1 = Point(l2[0], l2[1]);
			Point l2p2 = Point(l2[2], l2[3]);

			// find equation y=mx+c
			// where m = slope, c = intercept
			float l1Slope = (float)(l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
			float l2Slope = (float)(l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);

			if (l1Slope == l2Slope) continue;

			float l1Inter = l1p1.y - l1Slope * l1p1.x;
			float l2Inter = l2p1.y - l2Slope * l2p1.x;

			float interX = (l2Inter - l1Inter) / (l1Slope - l2Slope);
			float interY = (l1Slope * interX + l1Inter);

			Point interPoint = Point((int)interX, (int)interY);

			if (interPoint.x > frameSize.width || interPoint.x < 0 || interPoint.y > frameSize.height || interPoint.y < 0) continue;

			interPoints.push_back(interPoint);
		}
	}

	if (interPoints.size() < 1) return Point(-1, -1);

	Point vanishPoint = Point(0, 0);

	for (size_t i = 0; i < interPoints.size(); i++)
	{
		Point interPoint = interPoints[i];
		vanishPoint.x += interPoint.x;
		vanishPoint.y += interPoint.y;
	}

	vanishPoint.x /= interPoints.size();
	vanishPoint.y /= interPoints.size();

	return vanishPoint;
}

vector<Rect> RoadDetection::getVehicles(Mat frame)
{
	Mat equalizedFrame;
	CascadeClassifier vehiclesCascade;
	vector<Rect> vehicles;

	frame.copyTo(equalizedFrame);
	cvtColor(equalizedFrame, equalizedFrame, CV_BGR2GRAY);
	equalizeHist(equalizedFrame, equalizedFrame);

	vehiclesCascade.load("../Assets/cars.xml");
	vehiclesCascade.detectMultiScale(equalizedFrame, vehicles, cascadeScale, cascadeMinNeighbors, 0 | CASCADE_SCALE_IMAGE, Size(cascadeMinSize, cascadeMaxSize));

	return vehicles;
}