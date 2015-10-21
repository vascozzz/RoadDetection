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
	Mat originalImg, blurredImg, contoursImg, contoursInvImg, resultImg, houghImg;
	int houghVote = 200;
	vector<Vec2f> lines;

	// save a copy of the original image
	original.copyTo(originalImg);

	// attempt noise removal
	GaussianBlur(originalImg, blurredImg, Size(3, 3), 0, 0);

	// canny edge detection (should check thresh values), inverted
	Canny(blurredImg, contoursImg, 50, 350);
	threshold(contoursImg, contoursInvImg, 128, 255, THRESH_BINARY_INV);

	// hough voting process
	while (lines.size() < 5 && houghVote > 0){
		HoughLines(contoursImg, lines, 1, CV_PI / 180, houghVote, 0, 0);
		houghVote -= 5;
	}

	resultImg = Mat(contoursImg.rows, contoursImg.cols, CV_8U, Scalar(255));
	originalImg.copyTo(resultImg);

	// display hough result
	vector<Vec2f>::const_iterator it = lines.begin();
	houghImg = Mat(originalImg.size(), CV_8U, Scalar(0));

	while (it != lines.end())
	{
		float rho = (*it)[0];
		float theta = (*it)[1];

		// could make use of angle to eliminate some lines
		// if (theta < CV_PI / 20. || theta > 19. * CV_PI / 20.){}
		Point pt1(rho / cos(theta), 0);
		Point pt2((rho - resultImg.rows * sin(theta)) / cos(theta), resultImg.rows);

		line(resultImg, pt1, pt2, Scalar(255), 8);
		line(houghImg, pt1, pt2, Scalar(255), 8);

		++it;
	}

	// display
	namedWindow("input");
	namedWindow("contours");
	namedWindow("path");

	imshow("input", originalImg);
	imshow("contours", contoursInvImg);
	imshow("path", resultImg);
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