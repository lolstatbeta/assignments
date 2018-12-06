#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram�� �׽��ϴ�. 

					/** your code here! **/
					histogram[inputMat.at<uchar>(y, x)]++;
					// hint 1 : for loop �� �̿��ؼ� cv::Mat ��ȸ �� (1ä���� ���) 
					// inputMat.at<uchar>(y, x)�� ���� �����Ϳ� ������ �� �ֽ��ϴ�. 
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

			// Todo : hs 2���� ������׷��� ����ϴ� �Լ��� �ۼ��մϴ�. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			std::vector<cv::Mat> channels;
			split(srcMat, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];
			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : ����ȭ�� h,s ���� ��� histogram�� ���� ���մϴ�. 
					int quanH = UTIL::quantize(mat_h.at<uchar>(y, x));
					int quanS = UTIL::quantize(mat_s.at<uchar>(y, x));
					/** your code here! **/
					outputProb.at<double>(y, x) = UTIL::h_r(model_hist, input_hist, quanH, quanS);
					// hint 1 : UTIL::quantize()�� �̿��ؼ� srtMat�� ���� ����ȭ�մϴ�. 
					// hint 2 : UTIL::h_r() �Լ��� �̿��ؼ� outputPorb ���� ����մϴ�. 
				}
			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2���� ������׷��� �׽��ϴ�. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : ����ȭ�� h,s ���� ��� histogram�� ���� ���մϴ�. 

					/** your code here! **/
					int quanH = UTIL::quantize(mat_h.at<uchar>(y, x));
					int quanS = UTIL::quantize(mat_s.at<uchar>(y, x));
					// hint 1 : ����ȭ �� UTIL::quantize() �Լ��� �̿��ؼ� mat_h, mat_s�� ���� ����ȭ��ŵ�ϴ�. 
					histogram[quanH][quanS]++;
				}
			}

			// ������׷��� (hsv.rows * hsv.cols)���� ����ȭ�մϴ�. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram�� �ִ� ������ ��ȸ�ϸ� (hsv.rows * hsv.cols)���� ����ȭ�մϴ�. 
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

			// h^ ��� 
			for (int j = 0; j < 256; j++) {
				Histogram_Normal[j] = (double)inputHistogram[j] / (double)(inputMat.rows * inputMat.cols);
			}

			double u = 0.0;
			for (int j = 0; j < 256; j++) {
				u += (j*Histogram_Normal[j]);
			}

			// t>0 --> w0, u0, u1, between_Variance ���
			for (int j = 0; j < 256; j++) {
				if (j == 0)
				{
					// t=0 �ʱⰪ
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
