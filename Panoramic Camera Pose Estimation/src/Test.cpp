#include <iostream>  
#include <string>  
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846
// Satellite Image Setting
int GrdImgWidth = 2048;
int SatImgWidth = 512;
IplImage* ImgSat = 0;
IplImage* ImgGrd = 0;

// Data structure
// points on 2D plane and the relative angle
struct XYAngle
{
	// x, y and angle in cartesian coordiante( x: right is positive; y: upper is positive, unit of angle is in radius
	double x=0, y=0;
	double angle;
};

// Vectors used to store variables, every time the estimation process finished, all the elements in vector will be eliminated.
std::vector<double> vecX, vecY, vecAngle;
std::vector<XYAngle> vecXYTheta; //angle in vecXYTheta is cosine value

//on_mouse functions used to record the coordinate of point in Sat and Grd image, XY-Sat and Ori-Grd
void on_mouse_XY( int event, int x, int y, int flags, void* ustc)  
{  
    if( event == CV_EVENT_LBUTTONDOWN )  
    {
		//Tranfer UV to Cartesian coordinate
		vecX.push_back((double)x);
		vecY.push_back(SatImgWidth - (double)y);
		CvPoint pt = cvPoint(x,y);
		cvCircle(ImgSat, pt, 2, cvScalar(0, 255, 0, 0) ,CV_FILLED, CV_AA, 0);
        cvShowImage( "ImgSat", ImgSat);
    }   
}  

void on_mouse_Ori( int event, int x, int y, int flags, void* ustc)  
{  
    if( event == CV_EVENT_LBUTTONDOWN )  
    {
		//Tranfer oritation to Cartesian coordinate(in radius)
		double angle = (double)x / GrdImgWidth * (2 * PI) ;
		vecAngle.push_back(angle);
		CvPoint pt = cvPoint(x,y);
		cvCircle(ImgGrd, pt, 2, cvScalar(0, 255, 0, 0) ,CV_FILLED, CV_AA, 0);
        cvShowImage( "ImgGrd", ImgGrd);
    }   
}  

// Click function aggregate on_mouse functions and use 'isXY' to distinguish is XY or Ori
void Click(const char* _imgSatPath, const char* _imgGrdPath)
{
	ImgSat = cvLoadImage(_imgSatPath,1);
	ImgGrd = cvLoadImage(_imgGrdPath,1);
    cvNamedWindow("ImgSat",1);
	cvNamedWindow("ImgGrd",1);
	cvSetMouseCallback( "ImgSat", on_mouse_XY, 0 );
	cvSetMouseCallback( "ImgGrd", on_mouse_Ori, 0 );
    cvShowImage("ImgSat", ImgSat);  
	cvShowImage("ImgGrd", ImgGrd);
    cvWaitKey(0);
    cvDestroyAllWindows();  
    cvReleaseImage(&ImgSat);
	cvReleaseImage(&ImgGrd);
	return;
}

//cost function model
struct COST
{
	COST( double x_1, double y_1, double x_i, double y_i, double cosTheta ): _x_1(x_1), _y_1(y_1), _x_i(x_i), _y_i(y_i), _cosTheta(cosTheta) {}
	template <typename T>
	bool operator() (
		const T* const xy,
		T* residual ) const
	{
		residual[0] = T(_cosTheta) 
					  - 
					  ( (T(_x_1) - xy[0]) * (T(_x_i) - xy[0]) + (T(_y_1) - xy[1]) * (T(_y_i) - xy[1]) ) 
					  /
					  (
						ceres::sqrt(ceres::pow((T(_x_1) - xy[0]), 2) + ceres::pow((T(_y_1) - xy[1]), 2))
					  * ceres::sqrt(ceres::pow((T(_x_i) - xy[0]), 2) + ceres::pow((T(_y_i) - xy[1]), 2))
					  );
		return true;
	}
	const double _x_1, _y_1, _x_i, _y_i, _cosTheta;
};

//given the Img path and return the estimated x, y, and orientation(north is 0 degree)
XYAngle estimate(int index, string satImgPath, string grdImgPath)
{
	XYAngle result;
	string _satImgPath = satImgPath;
	string _grdImgPath = grdImgPath;
	Click(_satImgPath.c_str(), _grdImgPath.c_str());
	
	// Check
	if(vecX.size() <= 2|| vecX.size() != vecAngle.size())
	{
		cout << "Invalid number of xy and angle!" << endl;
		return result;
	}
	
	// build the dataset
	XYAngle firstPoint;
	firstPoint.x = vecX.at(0);
	firstPoint.y = vecY.at(0);
	firstPoint.angle = vecAngle.at(0);
	vecXYTheta.push_back(firstPoint);
	for(int i = 1; i < vecX.size(); i++)
	{
		XYAngle temp;
		temp.x = vecX.at(i);
		temp.y = vecY.at(i);
		temp.angle = std::cos(fabs(vecAngle.at(i) - vecXYTheta.at(0).angle));
		vecXYTheta.push_back(temp);
	}
	
	//Initial value
	double xy[2] = {256.0, 256.0};
	
	//build the least square problem
	ceres::Problem problem;
	for(int i = 1; i < vecXYTheta.size(); i++)
	{
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<COST, 1, 2>
			(
				new COST(vecXYTheta.at(0).x, vecXYTheta.at(0).y, vecXYTheta.at(i).x, vecXYTheta.at(i).y, vecXYTheta.at(i).angle)
			),
			nullptr,
			xy
			);
	}
	
	//solver settings
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	
	// solving
	ceres::Solver::Summary summary;
	cout << "-------------------------" << endl << "Ready to solve: " << endl;
	ceres::Solve(options, &problem, &summary);
    // result
	cout << "-------------------------" << endl << "Results: " << endl;
    cout<<summary.BriefReport() <<endl;
	cout << "-------------------------" << endl;
    cout<<"estimated x, y = ";
    for ( auto x:xy ) cout<<x<<" ";
    cout<<endl;
	
	//Estimate the orientation
	double oriSign = atan2((vecXYTheta.at(0).y - xy[1]), (vecXYTheta.at(0).x - xy[0]));
	//Since atan2 return value in region [-Pi, Pi], we need tansfer it to [0, 2*PI]
	if(oriSign >= -PI && oriSign <= PI/2)
	{
		oriSign = - oriSign + PI/2;
	}
	else
	{
		oriSign = - oriSign + 2.5 * PI;
	}
	double ori = oriSign + PI - vecAngle.at(0);
	if(ori > 2*PI)
	{
		ori = ori - 2 * PI;
	}
	cout << "-------------------------" << endl;
	cout << "The orientation of car: " << ori * 180 / PI << "Degree" << endl;
	
	//save the image
	//Ground truth
	Mat srcImg = imread(satImgPath);
	Point GroundTruth(SatImgWidth / 2, SatImgWidth / 2);
    circle(srcImg, GroundTruth, 5, Scalar(0, 255, 255), -1);
	
	//Visualize the estimated point
	Point estimatePosition(xy[0], SatImgWidth - xy[1]);
    circle(srcImg, estimatePosition, 5, Scalar(255, 255, 0), -1);
	
	imshow("least square", srcImg);
	waitKey(0);
	imwrite("./" + std::to_string(index) + ".jpg", srcImg);
	
	//eliminate all the elements in vectors and prepare for the next time usage
	vecX.clear();
	vecY.clear();
	vecAngle.clear();
	vecXYTheta.clear(); //angle in vecXYTheta is cosine value
	
	//return estimated value
	result.x = xy[0];
	result.y = xy[1];
	result.angle = ori * 180 / PI;
	
	return result;
	
}

//Get the image path
vector< std::string > ImgPath2Vec(std::string csvPath, int _index)
{
	cout << "Ready to open file..." << endl;
	vector<std::string> vecImgPath;
	ifstream file;
	file.open(csvPath, ifstream::in);
	if(!file)
	{
		cout<< "Cannot open file!" << endl;
		return vecImgPath;
	}

	string line;
	size_t mark = 0;
	size_t markTemp = 0;
	vector<int> vecMarkIndex;
	
	while (!file.eof())
	{
		getline(file,line);
		if(line.size() == 0) continue;
		
		//get mark index
		vector<int> vecMarkIndex;
		mark = line.find(',', 0);
		vecMarkIndex.push_back(mark);
		markTemp = mark;
		while(mark < line.size())
		{
			mark = line.find(',', markTemp + 1);
			vecMarkIndex.push_back(mark);
			markTemp = mark;
		}
		
		//get Azimuth
		string path = line.substr(vecMarkIndex[_index-1]+1, vecMarkIndex[_index]-vecMarkIndex[_index-1]-1);
		cout << "Image path: " << path << endl;
		vecImgPath.push_back(path);
	}

	file.close();
	cout << "File closed. " << endl;
	return vecImgPath;
}

int main()  
{
	std::string imgRoot = "/media/junbo/Elements/CVM_Dataset";
	std::string csvPath = "/media/junbo/Elements/CVM_Dataset/Accurate Localization/results.csv";
	vector<std::string> vecSatImgPath = ImgPath2Vec("/media/junbo/Elements/CVM_Dataset/Accurate Localization/information.csv", 0);
	vector<std::string> vecGrdImgPath = ImgPath2Vec("/media/junbo/Elements/CVM_Dataset/Accurate Localization/information.csv", 1);
	
	//Write csv file
	ofstream outFile;
	outFile.open(csvPath, ios::out);
	
	for(int i = 28; i < 29; i++)
	{
		XYAngle result = estimate(i, imgRoot + "/" + vecSatImgPath.at(i), imgRoot + vecGrdImgPath.at(i));
		outFile << result.x << "," << result.y << "," << result.angle << endl;
	}
	outFile.close();
    return 0;  
}  