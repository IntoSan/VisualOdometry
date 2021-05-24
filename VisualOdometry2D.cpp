#include <iostream>
#include <iomanip>
#include <fstream>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace Eigen;
// Flag for Initialization
bool initState = false;
bool ready = false;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void Initialization(const Mat &im, vector<KeyPoint> &keypoints, Mat &descriptors);

void find_feature_matches(const Mat &im,
  vector<KeyPoint> &lastKeypoints,
  vector<KeyPoint> &currentKeypoints,
  Mat &lastDescriptors, Mat &currentDescriptors, 
  vector<DMatch> &matches);


void pose_estimation_2d2d(std::vector< cv::KeyPoint > &lastKeypoints, std::vector< cv::KeyPoint > &currentKeypoints, std::vector< cv::DMatch > &matches, cv::Mat& R, cv::Mat& t);

void DrawTrajectory(cv::Mat& R, cv::Mat& t, Isometry3d &T, pangolin::OpenGlMatrix& M);

int main(int argc, char **argv) {
    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[1]), vstrImageFilenames, vTimestamps);
    
    // Number images
    int nImages = vstrImageFilenames.size();
    
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    
    // Points and descriptors
    vector<KeyPoint> lastKeypoints;
    Mat lastDescriptors;
    vector<KeyPoint> currentKeypoints;
    Mat currentDescriptors;
    vector<DMatch> matches;
    
    //Pangolin window
    
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    
    // Main loop
    Mat im;
    Isometry3d T = Isometry3d::Identity();
    pangolin::OpenGlMatrix M;
    M.SetIdentity();
    
    
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = imread(vstrImageFilenames[ni],IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];
        
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }
        
        if (initState == false)
        {
            Initialization(im, lastKeypoints, lastDescriptors);
        }
        else
        {
            find_feature_matches(im, lastKeypoints, currentKeypoints, lastDescriptors, currentDescriptors, matches);
            Mat R, t;
            pose_estimation_2d2d(lastKeypoints, currentKeypoints, matches, R, t);
            DrawTrajectory(R, t, T, M);
            
            // Replacement lastKeypoints and lastDescriptors
            lastKeypoints = currentKeypoints;
            lastDescriptors = currentDescriptors;
        }
        
    
    }
    return 0;
}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

void Initialization(const Mat &im, vector<KeyPoint> &keypoints, Mat &descriptors)
{
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    
    // detect Oriented FAST
    detector->detect(im, keypoints);
    
    // compute BRIEF descriptor
    descriptor->compute(im, keypoints, descriptors);
    
    // switch initialization flag
    initState = true;
}

void find_feature_matches(const cv::Mat& im, vector<KeyPoint>& lastKeypoints, vector<KeyPoint>& currentKeypoints, Mat &lastDescriptors, Mat &currentDescriptors, vector<DMatch>& matches)
{
    
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    //Computing descriptors for current image
    
    // detect Oriented FAST
    detector->detect(im, currentKeypoints);
    
    // compute BRIEF descriptor
    descriptor->compute(im, currentKeypoints, currentDescriptors);
    
    // use Hamming distance to match the features
    matcher->match(lastDescriptors, currentDescriptors, matches);
    
    // sort and remove the outliers
    
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    
    // remove the bad matching
    for (int i = 0; i < currentDescriptors.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(matches[i]);
    }}
    
    // Showing image with features
        Mat outIm;
        drawKeypoints(im, currentKeypoints, outIm);
        imshow("Sequence 06", outIm);
        waitKey(1);    
    
}

void pose_estimation_2d2d(vector<cv::KeyPoint> &lastKeypoints, vector<cv::KeyPoint> &currentKeypoints, vector<cv::DMatch> &matches, cv::Mat& R, cv::Mat& t)
{
    
    Mat K = (Mat_<double>(3, 3) << 707.1, 0, 601.9, 0, 707.1, 183.1, 0, 0, 1);
    
    // Convert the matching point to the form of vector<Point2f>
    vector<Point2f> points1;
    vector<Point2f> points2;
    
    for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(lastKeypoints[matches[i].queryIdx].pt);
    points2.push_back(currentKeypoints[matches[i].trainIdx].pt);
  }
  
    // Calculate essential_matrix
    Point2d principal_point(601.9, 183.1);
    double focal_length = 707;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    
    // Calculate pose of camera
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    //cout << "R is " << endl << R << endl;
    //cout << "t is " << endl << t << endl;
    
    
}

void DrawTrajectory(cv::Mat& R, cv::Mat& t, Isometry3d& T, pangolin::OpenGlMatrix& M)
{
    Matrix3d rotation_matrix;
    cv2eigen(R, rotation_matrix);
    
    Vector3d translation_vector;
    cv2eigen(t, translation_vector);
    
    Quaterniond q =  Quaterniond(rotation_matrix);
    Isometry3d Twr(q);

    Twr.pretranslate(translation_vector);
    
    T = T * Twr;
    //cout << "rotation_matrix is " << endl << rotation_matrix << endl;
    //cout << "translation_vector is " << endl << translation_vector << endl;
    M.m[0] = T(0,0);
    M.m[1] = T(1,0);    
    M.m[2] = T(2,0);
    M.m[3] = T(3,0);
    
    M.m[4] = T(0,1);
    M.m[5] = T(1,1);
    M.m[6] = T(2,1);
    M.m[7] = T(3,1);
    
    M.m[8] = T(0,2);
    M.m[9] = T(1,2);
    M.m[10] = T(2,2);
    M.m[11] = T(3,2);
    
    M.m[12] = T(0,3);
    M.m[13] = T(1,3);
    M.m[14] = T(2,3);
    M.m[15] = T(3,3);
    
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, 10, 1, 0,0,0,0.0, 1.0, 0.0)
    );
    /*else {
    s_cam.Follow(M);
    }*/
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
        
    glClear(GL_DEPTH_BUFFER_BIT);
    
        
    d_cam.Activate(s_cam);   
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);   
    
    
   
      // Drawing axes 
      Vector3d Ow = T.translation();
      Vector3d Xw = T * (-0.4 * Vector3d(1, 0, 0));
      Vector3d Yw = T * (-0.4 * Vector3d(0, 1, 0));
      Vector3d Zw = T * (-0.4 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
      
      pangolin::FinishFrame();
      usleep(5000);   // sleep 5 ms    
    

    
}
