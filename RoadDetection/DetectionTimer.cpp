#include "DetectionTimer.h"

DetectionTimer::DetectionTimer()
{
	initialTime = 0;
}

void DetectionTimer::start()
{
	initialTime = (double)cvGetTickCount();
}

double DetectionTimer::getElapsed()
{
	double elapsed = (double)cvGetTickCount() - initialTime;
	return elapsed / ((double)cvGetTickFrequency() * 1000.);
}