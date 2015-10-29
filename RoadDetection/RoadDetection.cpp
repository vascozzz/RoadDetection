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

void RoadDetection::processImage()
{
	Mat originalFrame, tmpFrame, grayFrame, blurredFrame, contoursFrame, houghFrame, sectionFrame, drawingFrame;
	vector<Line> houghLines, houghBestLines, houghProbLines;
	Point vanishingPoint;
	vector<Point> roadShape;
	int vanishHeight;

	// save a copy of the original frame
	original.copyTo(originalFrame);

	// smooth and remove noise
	GaussianBlur(originalFrame, blurredFrame, Size(blurKernel, blurKernel), 0, 0);

	// edge detection (canny, inverted)
	Canny(blurredFrame, contoursFrame, cannyLowThresh, cannyHighThresh);
	threshold(contoursFrame, contoursFrame, 128, 255, THRESH_BINARY);

	// hough transform lines
	houghFrame = Mat(originalFrame.size(), CV_8U, Scalar(0));
	houghLines = getHoughLines(contoursFrame, true);

	// vanishing point
	vanishingPoint = getVanishingPoint(houghLines, originalFrame.size());
	vanishHeight = vanishingPoint.y;

	// section frame (below vanishing point)
	contoursFrame.copyTo(tmpFrame);
	sectionFrame = tmpFrame(CvRect(0, vanishingPoint.y, contoursFrame.cols, contoursFrame.rows - vanishingPoint.y));

	// re-apply hough transform to section frame
	houghLines = getHoughLines(sectionFrame, true);
	houghLines = getFilteredLines(houghLines);

	// best line matches
	houghBestLines = getMainLines(houghLines);

	if (houghBestLines.size() >= 2)
	{
		vanishingPoint = getLineIntersection(houghBestLines[0], houghBestLines[1]);

		// get road shape
		roadShape = getRoadShape(sectionFrame, houghBestLines[0], houghBestLines[1], vanishingPoint);
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

	// road shape

	int lineType = 8;
	Point shapePoints[1][3];

	for (int i = 0; i < roadShape.size(); i++)
		shapePoints[0][i] = roadShape[i];

	const Point* ppt[1] = { shapePoints[0] };
	int npt[] = { 3 };

	fillPoly(drawingFrame, ppt, npt, 1, Scalar(0, 255, 0), lineType);

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

Mat RoadDetection::processVideo(Mat rawFrame)
{
	Mat originalFrame, tmpFrame, grayFrame, blurredFrame, contoursFrame, houghFrame, sectionFrame, drawingFrame;
	vector<Line> houghLines, houghMainLines, houghProbLines;
	Point vanishingPoint;
	vector<Rect> vehicles;
	int vanishHeight;

	// convert to grayscale
	cvtColor(rawFrame, grayFrame, CV_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);

	// smooth and remove noise
	GaussianBlur(grayFrame, blurredFrame, Size(blurKernel, blurKernel), 0);

	// edge detection (canny, inverted)
	Canny(blurredFrame, contoursFrame, cannyLowThresh, cannyHighThresh);
	threshold(contoursFrame, contoursFrame, 128, 255, THRESH_BINARY);

	// hough transform
	houghLines = getHoughLines(contoursFrame, true);

	// vanishing point
	vanishingPoint = getVanishingPoint(houghLines, rawFrame.size());
	vanishHeight = vanishingPoint.y;

	// section frame (below vanishing point)
	sectionFrame = contoursFrame(CvRect(0, vanishHeight, contoursFrame.cols, contoursFrame.rows - vanishHeight));

	// re-apply hough transform to section frame
	houghLines = getHoughLines(sectionFrame, true);

	// best line matches
	houghMainLines = getMainLines(houghLines);

	if (houghMainLines.size() >= 2)
	{
		Point intersection = getLineIntersection(houghMainLines[0], houghMainLines[1]);

		if (intersection.x > 0 && intersection.y > 0)
		{
			vanishingPoint = intersection;
			vanishHeight += vanishingPoint.y;
		}
	}

	rawFrame.copyTo(tmpFrame);
	drawingFrame = tmpFrame(CvRect(0, vanishHeight, contoursFrame.cols, contoursFrame.rows - vanishHeight));

	// best lines
	for (size_t i = 0; i < houghMainLines.size(); i++)
	{
		Point pt1 = houghMainLines[i].pt1;
		Point pt2 = houghMainLines[i].pt2;

		line(drawingFrame, pt1, pt2, Scalar(20, 125, 255), 2, CV_AA);
	}

	// combine drawing frame with original image
	for (int i = vanishingPoint.y; i < drawingFrame.rows; i++)
	{
		for (int j = 0; j < drawingFrame.cols; j++)
		{
			Vec3b pixel = drawingFrame.at<Vec3b>(i, j);
			rawFrame.at<Vec3b>(i + vanishHeight, j) = pixel;
		}
	}

	// vanish point
	vanishingPoint.y += vanishHeight;
	circle(rawFrame, vanishingPoint, 15, Scalar(20, 125, 255), -1, CV_AA);

	// vehicles
	vehicles = getVehicles(drawingFrame);

	for (size_t i = 0; i < vehicles.size(); i++)
	{
		vehicles[i].y += vanishingPoint.y;
		rectangle(rawFrame, vehicles[i], Scalar(255, 0, 0), 1, CV_AA);
	}

	return rawFrame;
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

	// find equation y = mx + c for intersection point between lines, where m = slope, c = intercept
	float l1Slope = l1.slope;
	float l2Slope = l2.slope;

	if (l1Slope == l2Slope)
	{
		return Point(0, 0);
	}

	float l1Inter = l1.intercept;
	float l2Inter = l2.intercept;

	float interX = (l2Inter - l1Inter) / (l1Slope - l2Slope);
	float interY = (l1Slope * interX + l1Inter);

	Point interPoint = Point((int)interX, (int)interY);

	return interPoint;
}

vector<Line> RoadDetection::getHoughLines(Mat frame, bool useMinimum)
{
	vector<Vec2f> lines;
	vector<Line> filteredLines;

	// if using  minimum, repeat hough process while lowering the voting until we have houghMinLines
	if (useMinimum)
	{
		int thresh = houghMinThresh;

		while ((int)lines.size() < houghMinLines && thresh > 0)
		{
			HoughLines(frame, lines, 1, CV_PI / 180, thresh, 0, 0);
			thresh -= 5;
		}
	}

	// otherwise, run hough process once using a default voting value
	else
	{
		HoughLines(frame, lines, 1, CV_PI / 180, houghDefaultThresh, 0, 0);
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

vector<Line> RoadDetection::getHoughProbLines(Mat frame)
{
	vector<Vec4i> lines;
	vector<Line> filteredLines;

	HoughLinesP(frame, lines, 1, CV_PI / 180, houghProbThresh, houghProbMinLineLength, houghProbMaxLineGap);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Point pt1 = Point(lines[i][0], lines[i][1]);
		Point pt2 = Point(lines[i][2], lines[i][3]);
		Line line = Line(pt1, pt2);

		if (line.angle > houghProbLowAngle && line.angle < houghProbHighAngle)
		{
			filteredLines.push_back(line);
		}
	}

	return filteredLines;
}

Point RoadDetection::getVanishingPoint(vector<Line> lines, Size frameSize)
{
	vector<Point> interPoints;
	vector<double> interAngles;

	double interAnglesSum = 0;
	Point vanishPoint = Point(0, 0);

	if (lines.size() <= 0)
	{
		return vanishPoint;
	}

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

			if (interPoint.x > frameSize.width || interPoint.x <= 0 || interPoint.y > frameSize.height || interPoint.y <= 0)
			{
				continue;
			}

			double interAngle = abs(l1.angle - l2.angle);
			interAnglesSum += interAngle;

			interPoints.push_back(interPoint);
			interAngles.push_back(interAngle);
		}
	}

	if (interPoints.size() < 1)
	{
		return vanishPoint;
	}

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
			}

			double angDiff = abs(l1.angle - l2.angle);

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
	{
		output.push_back(lines[lastIndex]);
	}	

	if (output.size() < lines.size())
	{
		return getFilteredLines(output);
	}

	return output;
}

vector<Line> RoadDetection::getMainLines(vector<Line> lines)
{
	vector<Line> mainLines;
	Line mainLine1, mainLine2;
	double bestAngle = -1.f;

	if ((int)lines.size() < 2)
	{
		return mainLines;
	}

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

			double angleDiff = abs(l1.angle - l2.angle);

			if (angleDiff > bestAngle)
			{
				bestAngle = angleDiff;
				mainLine1 = l1;
				mainLine2 = l2;
			}
		}
	}

	mainLines.push_back(mainLine1);
	mainLines.push_back(mainLine2);

	return mainLines;
}

vector<Point> RoadDetection::getRoadShape(Mat screen, Line l1, Line l2, Point inter)
{
	vector<Point> output;

	int screenSizeX = screen.cols;
	int screenSizeY = screen.rows;

	//simple side detection
	Line leftLine;
	Line rightLine;

	if (l1.slope > 0)
	{
		leftLine = l1;
		rightLine = l2;
	}
	else
	{
		leftLine = l2;
		rightLine = l1;
	}

	//Start filling point vector

	output.push_back(inter);

	cout << "Left Line: y = " << leftLine.slope << "x + " << leftLine.intercept << endl;
	cout << "Right Line: y = " << rightLine.slope << "x + " << rightLine.intercept << endl;

	// y = mx + b <=> screenSizeY = mx + b <=> x = (screenSizeY - b) / m
	Point leftPoint = Point((screenSizeY - leftLine.intercept) / leftLine.slope, screenSizeY);
	output.push_back(leftPoint);

	Point rightPoint = Point((screenSizeY - rightLine.intercept) / rightLine.slope, screenSizeY);
	output.push_back(rightPoint);

	return output;
}

vector<Rect> RoadDetection::getVehicles(Mat frame)
{
	Mat equalizedFrame;
	CascadeClassifier vehiclesCascade;
	vector<Rect> vehicles;

	cvtColor(frame, equalizedFrame, CV_BGR2GRAY);
	equalizeHist(equalizedFrame, equalizedFrame);

	vehiclesCascade.load("../Assets/cars.xml");
	vehiclesCascade.detectMultiScale(equalizedFrame, vehicles, cascadeScale, cascadeMinNeighbors, 0 | CASCADE_SCALE_IMAGE, Size(cascadeMinSize, cascadeMaxSize));

	return vehicles;
}

void onRoadDetectionTrackbarChange(int, void *userdata)
{
	RoadDetection roadDetector = *((RoadDetection*)userdata);
	roadDetector.processImage();
}

void RoadDetection::displayControls()
{
	namedWindow("Controls", WINDOW_FREERATIO);
	createTrackbar("Probabilistic Hough Threshold", "Controls", &houghProbThresh, 300, onRoadDetectionTrackbarChange, (void*)(this));
}