#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>

using namespace cv;

//Polynomial regression function
std::vector<double> fitPoly(std::vector<cv::Point> points, int n)
{
  //Number of points
  int nPoints = points.size();

  //Vectors for all the points' xs and ys
  std::vector<float> xValues = std::vector<float>();
  std::vector<float> yValues = std::vector<float>();

  //Split the points into two vectors for x and y values
  for(int i = 0; i < nPoints; i++)
  {
    xValues.push_back(points[i].x);
    yValues.push_back(points[i].y);
  }

  //Augmented matrix
  double matrixSystem[n+1][n+2];
  for(int row = 0; row < n+1; row++)
  {
    for(int col = 0; col < n+1; col++)
    {
      matrixSystem[row][col] = 0;
      for(int i = 0; i < nPoints; i++)
        matrixSystem[row][col] += pow(xValues[i], row + col);
    }

    matrixSystem[row][n+1] = 0;
    for(int i = 0; i < nPoints; i++)
      matrixSystem[row][n+1] += pow(xValues[i], row) * yValues[i];

  }

  //Array that holds all the coefficients
  double coeffVec[n+2]; // the "= {}" is needed in visual studio, but not in Linux

  //Gauss reduction
  for(int i = 0; i <= n-1; i++)
    for (int k=i+1; k <= n; k++)
    {
      double t=matrixSystem[k][i]/matrixSystem[i][i];

      for (int j=0;j<=n+1;j++)
        matrixSystem[k][j]=matrixSystem[k][j]-t*matrixSystem[i][j];

    }

  //Back-substitution
  for (int i=n;i>=0;i--)
  {
    coeffVec[i]=matrixSystem[i][n+1];
    for (int j=0;j<=n+1;j++)
      if (j!=i)
        coeffVec[i]=coeffVec[i]-matrixSystem[i][j]*coeffVec[j];

    coeffVec[i]=coeffVec[i]/matrixSystem[i][i];
  }

  //Construct the vector and return it
  std::vector<double> result = std::vector<double>();
  for(int i = 0; i < n+1; i++)
    result.push_back(coeffVec[i]);
  return result;
}

//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
cv::Point pointAtX(std::vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
    y += pow(x, i) * coeff[i];
  return cv::Point(x, y);
}


// -------------------------------------------------------------------------------------------------------



int main(int argc, char* argv[])
{
    Mat img;
    img = imread(argv[1], IMREAD_COLOR);

    // Check for failure
    if (img.empty()) {
        printf("Failed to load image '%s'\n", argv[1]);
        return -1;
    }

    Mat grayImg;
    if (img.channels() > 1) {
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    } else {
        grayImg = img.clone(); 
    }

    // Blur size
    int blurSize = 1;


    if (blurSize % 2 == 0) {
        blurSize++; 
    }

    // Apply Gaussian blur to the grayscale image
    Mat blurredImg;
    GaussianBlur(grayImg, blurredImg, Size(blurSize, blurSize), 4);

    // Applys Canny edge detection
    Mat edges;
    double lowerThreshold = 30; 
    double upperThreshold = 250;
    Canny(blurredImg, edges, lowerThreshold, upperThreshold);


    imshow("Canny Edges", edges);
    imwrite("edges.jpg", edges);


    // Apply probabilistic Hough transformation
    std::vector<Vec4i> lines;
    double rho = 1; 
    double theta = CV_PI / 180; 
    int threshold = 30; 
    HoughLinesP(edges, lines, rho, theta, threshold);

    // Draw the detected lines on the original image
    Mat houghLinesImg = img.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(houghLinesImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    imshow("Hough Lines", houghLinesImg);
    imwrite("houghLinesImg.jpg", houghLinesImg);


    // Remove short lines
    std::vector<Vec4i> filteredShortLines;
    double minLengthThreshold = 20; 
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2)); 
        if (length >= minLengthThreshold) { 
            filteredShortLines.push_back(l); 
    }

    // Draw the original image with short lines removed
    Mat imgWithoutShortLines = img.clone();
    for (size_t i = 0; i <filteredShortLines.size(); i++) {
        Vec4i l = filteredShortLines[i];
        line(imgWithoutShortLines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA); 
    }

    imshow("Original Image with Short Lines Removed", imgWithoutShortLines);
    imwrite("imgWithoutShortLines.jpg", imgWithoutShortLines);
    

    

    // Remove vertical lines
    std::vector<Vec4i> filteredVerticalLines;
    double maxAngleThreshold = 30.0; 
    for (size_t i = 0; i < filteredShortLines.size(); i++) {
        Vec4i l = filteredShortLines[i];
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI; 
        if (std::abs(angle) < maxAngleThreshold || std::abs(angle) > 180  - maxAngleThreshold) { 
            filteredVerticalLines.push_back(l); 
        }
    }

    // Draw the original image with vertical lines removed
    Mat imgWithoutVerticalLines = img.clone();
    for (size_t i = 0; i < filteredVerticalLines.size(); i++) {
        Vec4i l = filteredVerticalLines[i];
        line(imgWithoutVerticalLines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA); 
    }

    imshow("Original Image with Vertical Lines Removed", imgWithoutVerticalLines);
    imwrite("imgWithoutVerticalLines.jpg", imgWithoutVerticalLines);


    // Find nearly horizontal lines' points
    std::vector<Point> horizontalLinePoints;
    double angleThreshold = 30.0; 
    for (size_t i = 0; i < filteredShortLines.size(); i++) {
        Vec4i l = filteredShortLines[i];
        double dangle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI; 
        if (std::abs(dangle) < angleThreshold || std::abs(dangle) > 180 - angleThreshold) { 
            horizontalLinePoints.push_back(Point(l[0], l[1])); 
            horizontalLinePoints.push_back(Point(l[2], l[3])); 
        }
    }

    std::vector<double> curveCoefficients = fitPoly(horizontalLinePoints, 2); 

    // Draw the curve on the original image (Horizon Line)
    for (int x = 0; x < img.cols; x++) {
        int y = 0;
        for (int i = 0; i < curveCoefficients.size(); i++) {
            y += curveCoefficients[i] * pow(x, i);
        }
        if (y >= 0 && y < img.rows) {
            line(img, Point(x, y), Point(x, y), Scalar(0, 255, 0), 2, LINE_AA); 
        }
    }

    imshow("Original Image with Curve", img);
    imwrite("horizon.jpg", img);


    waitKey(0);

    return 0;
}