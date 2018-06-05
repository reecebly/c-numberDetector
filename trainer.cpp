#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include<vector>

using namespace cv;
using namespace std;

const int MIN_AREA = 100;
const int IMG_WIDTH_RE = 20;
const int IMG_HEIGHT_RE = 30;

//DIP Project
//Project Milestone
//Reece Bly, Nicholas Butman, Steven Treacy

int main() {

	Mat trainNums, grayscale, blur, threshold, thresholdCopy, classInts, trainingImgFF;

	vector<vector<Point> > ptContours;
	vector<Vec4i> vHierarchy;

	//valid characters we accept, limited for now. So far we have mainly done numbers, we hope to expand this
	vector<int> valChar = { '0', '1', '2', '3', '4', '5', '6',
		'7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
		'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };

	trainNums = imread("finalSubmission.jpg"); //read in train image0

	if (trainNums.empty()) //if cant open image
	{
		cout << "error\n\n";
		return(0);
	}

	cvtColor(trainNums, grayscale, CV_BGR2GRAY);

	GaussianBlur(grayscale, blur, Size(5, 5), 0);

	adaptiveThreshold(blur, threshold, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	

	imshow("imgThresh", threshold);

	thresholdCopy = threshold.clone(); //make a copy of thresh

	findContours(thresholdCopy, ptContours, vHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	

	for (int i = 0; i < ptContours.size(); i++)
	{
		if (contourArea(ptContours[i]) > MIN_AREA) {
			Rect rectBound = cv::boundingRect(ptContours[i]);

			//darw red rect
			rectangle(trainNums, rectBound, Scalar(0, 0, 255), 2);

			Mat ROI = threshold(rectBound);

			Mat ROIRe;
			resize(ROI, ROIRe, Size(IMG_WIDTH_RE, IMG_HEIGHT_RE)); //resize

			imshow("matROI", ROI);
			imshow("matROIResized", ROIRe);
			imshow("imgTrainingNumbers", trainNums);

			int intChar = waitKey(0); //wait for press

			if (intChar == 27)
			{
				return(0); //exit when escape is pressed
			}

			else if (find(valChar.begin(), valChar.end(), intChar) != valChar.end())
			{
				classInts.push_back(intChar);
				Mat imageF;
				ROIRe.convertTo(imageF, CV_32FC1);
				Mat imgFlFl = imageF.reshape(1, 1);
				trainingImgFF.push_back(imgFlFl);

			}
		}
	}

	cout << "Training done\n\n";


	FileStorage classFS("classifications.xml", FileStorage::WRITE);  //open classifications

	if (classFS.isOpened() == false) //if it cannot be opened
	{
		cout << "error, unable to open file\n\n";
		return(0);
	}

	classFS << "classifications" << classInts; //put classifcations in        
	classFS.release();  //close file

						//put images to file
	FileStorage trainIMG("images.xml", FileStorage::WRITE);

	if (trainIMG.isOpened() == false) //if file cannot be opened
	{
		cout << "error, unable to open file\n\n";
		return(0);
	}

	trainIMG << "images" << trainingImgFF; //write to file       
	trainIMG.release();  //close file                                                

	return(0);
}
