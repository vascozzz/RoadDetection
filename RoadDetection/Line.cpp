#include "Line.h"

Line::Line(Point pt1, Point pt2) :pt1(pt1), pt2(pt2) 
{
	angle = getAngleBetweenPoints();
	slope = getSlope();
	intercept = getIntercept();
}

double Line::getAngleBetweenPoints()
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

float Line::getSlope()
{
	return (float)(pt2.y - pt1.y) / (pt2.x - pt1.x);
}

float Line::getIntercept()
{
	return pt1.y - slope * pt1.x;
}