#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include<sstream>
#include<windows.h>

//DIP Project
//Project Milestone
//Reece Bly, Nicholas Butman, Steven Treacy

using namespace cv;
using namespace std;

const int MIN_AREA = 100;
const int IMG_WIDTH_RE = 20;
const int IMG_HEIGHT_RE = 30;

class Contour
{
public:

	vector<Point> contourPT;  // variable for contour point
	Rect rectBound; // contour bounding rectangly
	float areaFloat;  //contour area


	bool contourVal()
	{
		//if contour is valid
		if (areaFloat < MIN_AREA) return false;
		return true;
	}


	static bool rectXPos(const Contour& contLeft, const Contour& contRight)
	{
		//sorts the contours
		return(contLeft.rectBound.x < contRight.rectBound.x);
	}

};


int main()
{
	
	Mat classInt;

	FileStorage classFS("classifications.xml", FileStorage::READ); //get file

	if (classFS.isOpened() == false) //if the file cannot be opened.
	{
		cout << "error, cant open file\n\n";
		return(0);
	}

	classFS["classifications"] >> classInt; //put classif into mat
	classFS.release(); //close the file                                        

	Mat trainingIMG;  //read img to be used for training

	FileStorage trainingIMGStore("images.xml", FileStorage::READ); //open the training img file

	if (trainingIMGStore.isOpened() == false)  //if the file cannot be opened.
	{
		cout << "error, cant open file\n\n";
		return(0);
	}

	trainingIMGStore["images"] >> trainingIMG; //put images into mat
	trainingIMGStore.release();  //close                                               

	Ptr<ml::KNearest>  kNearest(ml::KNearest::create()); //KNN

	kNearest->train(trainingIMG, ml::ROW_SAMPLE, classInt);
	
	//start video capture
	VideoCapture cap;
	if (!cap.open(0))
		return 0;

	
	for (;;)
	{
		vector<Contour> contoursD; //used for contours        
		vector<Contour> vContours;
		Mat testNums;
		cap >> testNums;
		//imshow("matTestingNumbers1", testNums);


		//Mat testNums = imread("test2.PNG");//read test nums

		if (testNums.empty()) //if cant open image
		{
			cout << "error cant open\n\n";
			return(0);
		}

		Mat grayscale, blur, threshold, thresholdCopy;

		cvtColor(testNums, grayscale, CV_BGR2GRAY);

		GaussianBlur(grayscale, blur, Size(5, 5), 0);

		adaptiveThreshold(blur, threshold, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);


		thresholdCopy = threshold.clone(); //make a copy of thres

		vector<vector<Point> > ptContours; //vector points    
		vector<Vec4i> hierarchy;

		findContours(thresholdCopy, ptContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


		for (int i = 0; i < ptContours.size(); i++) //creates list of contours
		{
			Contour contourD;
			contourD.contourPT = ptContours[i];
			contourD.rectBound = boundingRect(contourD.contourPT);
			contourD.areaFloat = contourArea(contourD.contourPT);
			contoursD.push_back(contourD);
		}

		for (int i = 0; i < contoursD.size(); i++) //for contours
		{
			if (contoursD[i].contourVal()) //if valid contour add it to list of valids
			{
				vContours.push_back(contoursD[i]);
			}
		}

		// sort contours from left to right
		sort(vContours.begin(), vContours.end(), Contour::rectXPos);

		string stringF;

		for (int i = 0; i < vContours.size(); i++) //for contours
		{
			//draws green rectangle around selection
			rectangle(testNums, vContours[i].rectBound, Scalar(0, 255, 0), 2);

			//ROI=region of interest
			Mat ROI = threshold(vContours[i].rectBound); //get roi of rect
			Mat ROIRe;
			resize(ROI, ROIRe, Size(IMG_WIDTH_RE, IMG_HEIGHT_RE));
			Mat ROIF;
			ROIRe.convertTo(ROIF, CV_32FC1); //converts to float, neccessary for KNN
			Mat ROIFFlat = ROIF.reshape(1, 1);
			Mat curChar(0, 0, CV_32F);
			kNearest->findNearest(ROIFFlat, 1, curChar);  //KNN
			float fltCurrentChar = (float)curChar.at<float>(0, 0);
			stringF = stringF + char(int(fltCurrentChar)); //build list of chars to display
		}

		waitKey(100);
		cout << "\n\n" << "numbers read = " << stringF << "\n\n"; //show characters
		imshow("matTestingNumbers", testNums);
		
		
	}

		waitKey(0);
		cap.release();
		return(0);
	
}
