# Road Detection

Developed for the class of Augmented Reality (EIC0104) at MIEIC, FEUP. 

> Road detection is a key requirement for unmanned guided vehicles as well as driver assistance. The main objective of this project is to develop a program for the automatic segmentation of structured roads, roads having clear edges and artificial lane markings, and for the detection of the road limits and the vanishing point. 

> The application here presented was fully developed in C++ using OpenCV (Open Source Computer Vision). The algorithm here proposed is divided in 10 individual steps, including edge detection (via application of the Canny Edge Detection algorithm), vanishing point calculation and lines extraction (relying on the Hough Transform technique).

Further info can be found in the [full report](Docs/report.pdf).

## Examples

![example](Docs/example2.png)
![example](Docs/example1.png)
![example](Docs/example3.png)
![example](Docs/example6.png)

Additional examples can be found in the Docs folder and in the [annex](Docs/annex.pdf).

## Usage Instructions

 The program can be ran by invoking it from the command line as follows:

```
RoadDetection.exe -method -filepath
```

The available methods are image(1) and video(2). The base path is *../Assets/.* Files should be moved to the Assets folder in order to tested. For example, in order to use the file *sample.png* inside the Assets folder, the program would be invoked as:

```
RoadDetection.exe 1 sample.png
```