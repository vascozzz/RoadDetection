#include "RoadDetection.h"

RoadDetection::RoadDetection(String filePath)
{
	original = imread(filePath);

	if (!original.data)
	{
		cout << "Unable to read from file. Now exiting..." << endl;
		exit(1);
	}
}

RoadDetection::RoadDetection(Mat frame)
{
	frame.copyTo(original);
}

void RoadDetection::setFile(String filePath)
{
	original = imread(filePath);

	if (!original.data)
	{
		cout << "Unable to read from file. Now exiting..." << endl;
		exit(1);
	}
}

void RoadDetection::setFile(Mat frame)
{
	frame.copyTo(original);
}

void RoadDetection::method1()
{
	Mat originalFrame, grayFrame, erodedFrame, dilatedFrame, morphFrame, resultFrame;

	// save a copy of the original frame
	original.copyTo(originalFrame);

	// convert to grayscale, threshold (if pixel > thresh then pixel = max, otherwise 0)
	cvtColor(originalFrame, grayFrame, CV_BGR2GRAY);
	threshold(grayFrame, grayFrame, 100, 255, THRESH_BINARY);

	// eroding process (picks local minimum pixel value from kernel; increases dark areas)
	erode(grayFrame, erodedFrame, Mat(), Point(2, 2), 7);

	// dilating process (picks local maximum pixel value from kernel; increases light areas)
	dilate(grayFrame, dilatedFrame, Mat(), Point(2, 2), 7);

	// threshold dilated image (if pixel > thresh then pixel = 0, otherwise max)
	threshold(dilatedFrame, dilatedFrame, 1, 50, THRESH_BINARY_INV);

	// combine both
	morphFrame = Mat(grayFrame.size(), CV_8U, Scalar(0));
	morphFrame = erodedFrame + dilatedFrame;

	// watershed (converting from unsigned 8-bit per pixel (0-255) to signed 32-bit for math operations, and then back for display)
	morphFrame.convertTo(resultFrame, CV_32S);
	watershed(originalFrame, resultFrame);
	resultFrame.convertTo(resultFrame, CV_8U);

	// display
	namedWindow("input");
	namedWindow("path");

	imshow("input", originalFrame);
	imshow("path", resultFrame);
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
	Mat originalFrame, tmpFrame, grayFrame, blurredFrame, contoursFrame, houghFrame, sectionFrame, drawingFrame;
	vector<Line> houghLines, houghBestLines, houghProbLines;
	Point vanishingPoint;
	int vanishHeight;

	// save a copy of the original frame
	original.copyTo(originalFrame);

	// convert to grayscale
	cvtColor(originalFrame, grayFrame, CV_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);

	// smooth and remove noise
	GaussianBlur(originalFrame, blurredFrame, Size(blurKernel, blurKernel), 0, 0);

	// edge detection (canny, inverted)
	Canny(blurredFrame, contoursFrame, cannyLowThresh, cannyHighThresh);
	threshold(contoursFrame, contoursFrame, 128, 255, THRESH_BINARY);

	// hough transform lines
	houghFrame = Mat(originalFrame.size(), CV_8U, Scalar(0));
	houghLines = getHoughLines(contoursFrame);

	// vanishing point
	vanishingPoint = getVanishingPoint(houghLines, originalFrame.size());
	vanishHeight = vanishingPoint.y;

	// section frame (below vanishing point)
	contoursFrame.copyTo(tmpFrame);
	sectionFrame = tmpFrame(CvRect(0, vanishingPoint.y, contoursFrame.cols, contoursFrame.rows - vanishingPoint.y));

	// re-apply hough transform to section frame
	houghLines = getHoughLines(sectionFrame);
	houghLines = getFilteredLines(houghLines);

	// best line matches
	houghBestLines = getMainLines(houghLines);

	if (houghBestLines.size() >= 2)
	{
		vanishingPoint = getLineIntersection(houghBestLines[0], houghBestLines[1]);
	}

	/* DRAWING */
	originalFrame.copyTo(tmpFrame);
	drawingFrame = tmpFrame(CvRect(0, vanishHeight, contoursFrame.cols, contoursFrame.rows - vanishHeight));

	// hough lines
	for (size_t i = 0; i < houghLines.size(); i++)
	{
		Point pt1 = houghLines[i].pt1;
		Point pt2 = houghLines[i].pt2;

		line(drawingFrame, pt1, pt2, Scalar(0, 0, 255), 2, CV_AA);
	}

	// best lines
	for (size_t i = 0; i < houghBestLines.size(); i++)
	{
		Point pt1 = houghBestLines[i].pt1;
		Point pt2 = houghBestLines[i].pt2;

		line(drawingFrame, pt1, pt2, Scalar(20, 125, 255), 2, CV_AA);
	}

	// vanishing point
	if (vanishingPoint.x > 0)
	{
		circle(drawingFrame, vanishingPoint, 15, Scalar(20, 125, 255), -1, CV_AA);
	}

	// combine drawing frame with original image
	for (int i = 0; i < drawingFrame.rows; i++)
	{
		for (int j = 0; j < drawingFrame.cols; j++)
		{
			Vec3b pixel = drawingFrame.at<Vec3b>(i, j);
			originalFrame.at<Vec3b>(i + vanishHeight, j) = pixel;
		}
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

double RoadDetection::getDistBetweenPoints(Point pt1, Point pt2)
{
	return sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));
}

Point RoadDetection::getPointAverage(Point pt1, Point pt2)
{
	return Point((pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2);
}

Point RoadDetection::getLineIntersection(Line l1, Line l2)
{
	Point l1p1 = l1.pt1;
	Point l1p2 = l1.pt2;
	Point l2p1 = l2.pt1;
	Point l2p2 = l2.pt2;

	// find equation y = mx + c for intersection point between lines
	// where m = slope, c = intercept
	float l1Slope = (float)(l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
	float l2Slope = (float)(l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);

	if (l1Slope == l2Slope)
	{
		return Point(-1, -1);
	}

	float l1Inter = l1p1.y - l1Slope * l1p1.x;
	float l2Inter = l2p1.y - l2Slope * l2p1.x;

	float interX = (l2Inter - l1Inter) / (l1Slope - l2Slope);
	float interY = (l1Slope * interX + l1Inter);

	Point interPoint = Point((int)interX, (int)interY);

	return interPoint;
}

vector<Line> RoadDetection::getHoughLines(Mat frame)
{
	vector<Vec2f> lines;
	vector<Line> filteredLines;

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

Line line = Line(pt1, pt2);
filteredLines.push_back(line);
		}
	}

	return filteredLines;
}

vector<Line> RoadDetection::getMainLines(vector<Line> lines)
{
	vector<Line> mainLines;
	Line mainLine1, mainLine2;
	double bestSlope = -1.f;

	for (size_t i = 0; i < lines.size() - 1; i++)
	{
		for (size_t j = i + 1; j < lines.size(); j++)
		{
			Line l1 = lines[i];
			Line l2 = lines[j];

			Point l1p1 = l1.pt1;
			Point l1p2 = l1.pt2;
			Point l2p1 = l2.pt1;
			Point l2p2 = l2.pt2;

			double angleLine1 = getAngleBetweenPoints(l1p1, l1p2);
			double angleLine2 = getAngleBetweenPoints(l2p1, l2p2);

			double slopeDiff = abs(angleLine1 - angleLine2);

			if (slopeDiff > bestSlope)
			{
				bestSlope = slopeDiff;
				mainLine1 = l1;
				mainLine2 = l2;
			}
		}
	}

	mainLines.push_back(mainLine1);
	mainLines.push_back(mainLine2);

	return mainLines;
}

vector<Line> RoadDetection::getHoughProbLines(Mat frame)
{
	vector<Vec4i> lines;
	vector<Line> filteredLines;

	HoughLinesP(frame, lines, 1, CV_PI / 180, houghProbThresh, houghProbMinLineLength, houghProbMaxLineGap);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Point pt1 = Point(lines[i][0], lines[i][1]);
		Point pt2 = Point(lines[i][2], lines[i][3]);

		double angle = getAngleBetweenPoints(pt1, pt2);

		if (angle > houghProbLowAngle && angle < houghProbHighAngle)
		{
			Line line = Line(pt1, pt2);
			filteredLines.push_back(line);
		}
	}

	return filteredLines;
}

vector<Line> RoadDetection::getFilteredLines(vector<Line> lines)
{
	vector<Line> output;
	vector<bool> linesToProcess(lines.size(), true);

	for (size_t i = 0; i < lines.size() - 1; i++)
	{
		if (!linesToProcess[i])
		{
			continue;
		}

		for (size_t j = i + 1; j < lines.size(); j++)
		{
			Line l1 = lines[i];
			Line l2 = lines[j];

			if (!linesToProcess[j])
			{
				continue;
				cout << "fuck " << j << " man" << endl;
			}

			double l1Angle = getAngleBetweenPoints(l1.pt1, l1.pt2);
			double l2Angle = getAngleBetweenPoints(l2.pt1, l2.pt2);

			double angDiff = abs(l1Angle - l2Angle);
			if (angDiff < maxLineAngleDiff)
			{
				double distpt1 = getDistBetweenPoints(l1.pt1, l2.pt1);
				double distpt2 = getDistBetweenPoints(l1.pt2, l2.pt2);

				if (distpt1 < maxLineDistDiff || distpt2 < maxLineDistDiff)
				{
					Line combinedLine = Line(getPointAverage(l1.pt1, l2.pt1), getPointAverage(l1.pt2, l2.pt2));

					output.push_back(combinedLine);
					linesToProcess[i] = false;
					linesToProcess[j] = false;
					break;
				}
			}

			if (j == (lines.size() - 1))
			{
				output.push_back(l1);
			}
		}
	}

	int lastIndex = lines.size() - 1;
	if (linesToProcess[lastIndex])
		output.push_back(lines[lastIndex]);

	if (output.size() < lines.size())
		return getFilteredLines(output);

	return output;
}

Point RoadDetection::getVanishingPoint(vector<Line> lines, Size frameSize)
{
	vector<Point> interPoints;
	vector<double> interAngles;
	double interAnglesSum = 0;

	for (size_t i = 0; i < lines.size() - 1; i++)
	{
		for (size_t j = i + 1; j < lines.size(); j++)
		{
			Line l1 = lines[i];
			Line l2 = lines[j];

			Point l1p1 = l1.pt1;
			Point l1p2 = l1.pt2;
			Point l2p1 = l2.pt1;
			Point l2p2 = l2.pt2;

			Point interPoint = getLineIntersection(l1, l2);

			if (interPoint.x > frameSize.width || interPoint.x < 0 || interPoint.y > frameSize.height || interPoint.y < 0)
			{
				continue;
			}

			// get angle between lines
			double angleLine1 = getAngleBetweenPoints(l1p1, l1p2);
			double angleLine2 = getAngleBetweenPoints(l2p1, l2p2);

			double interAngle = abs(angleLine1 - angleLine2);
			interAnglesSum += interAngle;

			interPoints.push_back(interPoint);
			interAngles.push_back(interAngle);
		}
	}

	Point vanishPoint = Point(0, 0);
	if (interPoints.size() < 1) vanishPoint;

	for (size_t i = 0; i < interPoints.size(); i++)
	{
		Point interPoint = interPoints[i];
		double interAngle = interAngles[i];
		double weight = interAngle / interAnglesSum;

		vanishPoint.x += cvRound(interPoint.x * weight);
		vanishPoint.y += cvRound(interPoint.y * weight);
	}

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

void onRoadDetectionTrackbarChange(int, void *userdata)
{
	RoadDetection roadDetector = *((RoadDetection*)userdata);
	roadDetector.detectAll();
}

void RoadDetection::displayControls()
{
	namedWindow("Controls", WINDOW_FREERATIO);
	createTrackbar("Probabilistic Hough Threshold", "Controls", &houghProbThresh, 300, onRoadDetectionTrackbarChange, (void*)(this));
}


// vehicles
// vehicles = getVehicles(originalFrame);
/*for (size_t i = 0; i < vehicles.size(); i++)
{
Point center(vehicles[i].x + vehicles[i].width / 2, vehicles[i].y + vehicles[i].height / 2);
ellipse(originalFrame, center, Size(vehicles[i].width / 2, vehicles[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 1, CV_AA, 0);
}*/