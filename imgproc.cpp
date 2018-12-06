#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram을 쌓습니다. 

					/** your code here! **/
					histogram[inputMat.at<uchar>(y, x)]++;
					// hint 1 : for loop 를 이용해서 cv::Mat 순회 시 (1채널의 경우) 
					// inputMat.at<uchar>(y, x)와 같이 데이터에 접근할 수 있습니다. 
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_64FC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// Todo : hs 2차원 히스토그램을 계산하는 함수를 작성합니다. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			std::vector<cv::Mat> channels;
			split(srcMat, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];
			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					int quanH = UTIL::quantize(mat_h.at<uchar>(y, x));
					int quanS = UTIL::quantize(mat_s.at<uchar>(y, x));
					/** your code here! **/
					outputProb.at<double>(y, x) = UTIL::h_r(model_hist, input_hist, quanH, quanS);
					// hint 1 : UTIL::quantize()를 이용해서 srtMat의 값을 양자화합니다. 
					// hint 2 : UTIL::h_r() 함수를 이용해서 outputPorb 값을 계산합니다. 
				}
			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 

					/** your code here! **/
					int quanH = UTIL::quantize(mat_h.at<uchar>(y, x));
					int quanS = UTIL::quantize(mat_s.at<uchar>(y, x));
					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
					histogram[quanH][quanS]++;
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram에 있는 값들을 순회하며 (hsv.rows * hsv.cols)으로 정규화합니다. 
					/** your code here! **/
					histogram[j][i] = histogram[j][i] / (hsv.rows * hsv.cols);
				}
			}
		}

		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int & threshold)
		{
			cv::Mat inputMat = src.getMat();
			dst.create(inputMat.size(), CV_8UC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					if (inputMat.at<uchar>(y, x) >= threshold)
						outputProb.at<uchar>(y, x) = 255;
					else
						outputProb.at<uchar>(y, x) = 0;
				}
			}

		}
		void thresh_otsu(cv::InputArray src, cv::OutputArray dst)
		{
			cv::Mat inputMat = src.getMat();
			dst.create(inputMat.size(), CV_8UC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			int inputHistogram[256] = { 0, };
			double Histogram_Normal[256] = { 0, };
			double w0[256] = { 0, };
			double u0[256] = { 0, };
			double u1[256] = { 0, };
			double between_V[256] = { 0, };

			calcHist(inputMat, inputHistogram);

			// h^ 계산 
			for (int j = 0; j < 256; j++) {
				Histogram_Normal[j] = (double)inputHistogram[j] / (double)(inputMat.rows * inputMat.cols);
			}

			double u = 0.0;
			for (int j = 0; j < 256; j++) {
				u += (j*Histogram_Normal[j]);
			}

			// t>0 --> w0, u0, u1, between_Variance 계산
			for (int j = 0; j < 256; j++) {
				if (j == 0)
				{
					// t=0 초기값
					w0[0] = Histogram_Normal[0];
					u0[0] = 0.0;
				}
				else {
					w0[j] = w0[j - 1] + Histogram_Normal[j];
					if (w0[j] == 0.0 || (1-w0[j]) == 0.0)
					continue;
					u0[j] = ((w0[j - 1] * u0[j - 1]) + (j*Histogram_Normal[j])) / w0[j];
					u1[j] = (u - (w0[j] * u0[j])) / (1 - w0[j]);
					between_V[j] = w0[j] * (1 - w0[j])*(u0[j] - u1[j])*(u0[j] - u1[j]);
				}
			}

			double max = 0.0;
			int threshold = 0;
			for (int i = 1; i < 256; i++) {
				if (between_V[i] > max) {
					max = between_V[i];
					threshold = i;
				}
			}

			thresh_binary(inputMat, outputProb, threshold);
		}
	}  // namespace IMG_PROC

}
