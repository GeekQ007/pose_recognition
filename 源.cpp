//
//  this sample demonstrates the use of pretrained openpose networks with opencv's dnn module.
//
//  it can be used for body pose detection, using either the COCO model(18 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
//
//  or the MPI model(16 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt
//
//  (to simplify this sample, the body models are restricted to a single person.)
//
//
//  you can also try the hand pose model:
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
//  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt
//

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;


// connection table, in the format [model_id][pair_id][from/to]
// please look at the nice explanation at the bottom of:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
const int POSE_PAIRS[3][20][2] = {
	{   // COCO body
		{ 1,2 },{ 1,5 },{ 2,3 },
		{ 3,4 },{ 5,6 },{ 6,7 },
		{ 1,8 },{ 8,9 },{ 9,10 },
		{ 1,11 },{ 11,12 },{ 12,13 },
		{ 1,0 },{ 0,14 },
		{ 14,16 },{ 0,15 },{ 15,17 }
	},
	{   // MPI body
		{ 0,1 },{ 1,2 },{ 2,3 },
		{ 3,4 },{ 1,5 },{ 5,6 },
		{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
		{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	},
	{   // hand
		{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
		{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
		{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
		{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
		{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
	} };

void drawBody(cv::Mat& frame, cv::Mat& result, double thresh);

int main()
{
	String modelTxt = "D:/opencv/CXK_POSE/model/pose_deploy_linevec.prototxt";
	String modelBin = "D:/opencv/CXK_POSE/model/pose_iter_440000.caffemodel";
	int W_in = 184;
	int H_in = 92;
	float thresh = 0.1;

	int backendId = cv::dnn::DNN_BACKEND_OPENCV;
	int targetId = cv::dnn::DNN_TARGET_CPU;

	// read the network model
	Net net = readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);

	VideoCapture cap("D:/opencv/CXK_POSE/cxk.mp4");
	if (!cap.isOpened()) {
		cerr << "open cam err." << endl;
		return 0;
	}
	VideoWriter writer("D:/opencv/CXK_POSE/cxk_pose.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
	while (true) {
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		resize(frame, frame, Size(640, 480));
		// send it through the network
		Mat inputBlob = blobFromImage(frame, 1.0 / 255, Size(W_in, H_in), (0, 0, 0), true, false);
		net.setInput(inputBlob);
		Mat result = net.forward();
		// the result is an array of "heatmaps", the probability of a body part being in location x,y
		drawBody(frame, result, thresh);

		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		writer.write(frame);

		imshow("OpenPose", frame);
		waitKey(1);
	}
	cap.release();
	writer.release();
	destroyAllWindows();
	return 0;
}


void drawBody(cv::Mat& frame, cv::Mat& result, double thresh)
{
	int midx, npairs;
	int nparts = result.size[1]/3;
	int H = result.size[2];
	int W = result.size[3];

	// find out, which model we have
	if (nparts == 19) {   // COCO body
		midx = 0;
		npairs = 17;
		nparts = 18; // skip background
	}
	else if (nparts == 16) {   // MPI body
		midx = 1;
		npairs = 14;
	}
	else if (nparts == 22) {   // hand
		midx = 2;
		npairs = 20;
	}
	else {
		cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
		return;
	}

	// find the position of the body parts
	vector<Point> points(22);
	for (int n = 0; n < nparts; n++) {
		// Slice heatmap of corresponding body's part.
		Mat heatMap(H, W, CV_32F, result.ptr(0, n));
		// 1 maximum per heatmap
		Point p(-1, -1), pm;
		double conf;
		minMaxLoc(heatMap, 0, &conf, 0, &pm);
		if (conf > thresh)
			p = pm;
		points[n] = p;
	}

	// connect body parts and draw it !
	float SX = float(frame.cols) / W;
	float SY = float(frame.rows) / H;
	for (int n = 0; n < npairs; n++) {
		// lookup 2 connected body/hand parts
		Point2f a = points[POSE_PAIRS[midx][n][0]];
		Point2f b = points[POSE_PAIRS[midx][n][1]];

		// we did not find enough confidence before
		if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
			continue;

		// scale to image size
		a.x *= SX; a.y *= SY;
		b.x *= SX; b.y *= SY;

		line(frame, a, b, Scalar(0, 200, 0), 2);
		circle(frame, a, 3, Scalar(0, 0, 200), -1);
		circle(frame, b, 3, Scalar(0, 0, 200), -1);
	}
}
