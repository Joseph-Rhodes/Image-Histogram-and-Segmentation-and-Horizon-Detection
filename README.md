# COMP27112 Visual Computing Lab 4 Report

## Overview
This repository contains the Lab 4 report for COMP27112 Visual Computing. The report includes implementations and results for tasks such as image histogram generation, thresholding, and horizon detection using OpenCV.

## How to View the Report
Click the link below to open the report in your web browser:

[View Report](./COMP27112_Lab4_Report.pdf)

Alternatively, you can manually open the `COMP27112_Lab4_Report.pdf` file using any PDF viewer.

## Contents
- **comp27112_lab4.pdf** – The original lab assignment document.
- **COMP27112_Lab4_Report.pdf** – The completed lab report.

## Technologies Used
- **Programming Language:** C++
- **Libraries:** OpenCV
- **Image Processing Techniques:** Histogram analysis, Thresholding, Edge Detection, Hough Transform

## Instructions for Running Code
To compile and run the code from the report:
1. Ensure you have OpenCV installed on your system.
2. Compile the C++ code using:
   ```sh
   g++ -o histogram histogram.cpp $(pkg-config --cflags --libs opencv4)
   ./histogram input_image.jpg
