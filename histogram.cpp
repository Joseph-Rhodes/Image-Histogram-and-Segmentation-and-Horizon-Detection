#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const int HIST_IMG_HEIGHT = 400;
const int HIST_IMG_WIDTH = 512;
const int BAR_WIDTH = 2;
const int GRID_LINES[] = {0, 64, 128, 192, 255}; 

void createHistogram(Mat& img, Mat& hist)
{
    long counts[256] = {};


	// Create a white image for the histogram
    hist = Mat(HIST_IMG_HEIGHT, HIST_IMG_WIDTH, CV_8UC1, Scalar(255));

    // Draw grid lines
    for (int i = 0; i < sizeof(GRID_LINES) / sizeof(GRID_LINES[0]); ++i) {
        int grid_line = GRID_LINES[i] * BAR_WIDTH;
        line(hist, Point(grid_line, 0), Point(grid_line, HIST_IMG_HEIGHT - 1), Scalar(200));
    }

    // Counts of each gray level
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            uchar pixel = img.at<uchar>(y, x);
            counts[pixel]++;
        }
    }

    // Max count
    long maxCount = 0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > maxCount) {
            maxCount = counts[i];
        }
    }

    // Bars of the histogram
    for (int i = 0; i < 256; ++i) {
        int barHeight = static_cast<int>((counts[i] * HIST_IMG_HEIGHT) / maxCount);

        if (barHeight > 0) {
            int startX = i * BAR_WIDTH;
            int endX = (i + 1) * BAR_WIDTH - 1;

            rectangle(hist, Point(startX, HIST_IMG_HEIGHT - barHeight),
                      Point(endX, HIST_IMG_HEIGHT - 1), Scalar(0), FILLED);
        }
    }
}

int main(int argc, char* argv[])
{
    Mat img;
    Mat hist;

    img = imread(argv[1], IMREAD_GRAYSCALE);

    // Check if the image was successfully loaded
    if (img.empty()) {
        printf("Failed to load image '%s'\n", argv[1]);
        return -1;
    }

    // Create image histogram
    createHistogram(img, hist);

    namedWindow("Histogram", WINDOW_NORMAL);
    imwrite("hist.jpg", hist);

    imshow("Histogram", hist);

    // Wait for a key press before quitting
    waitKey(0);

    return 0;
}
