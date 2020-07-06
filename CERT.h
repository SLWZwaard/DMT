#pragma once



#include <dlib/image_processing/shape_predictor.h>
#include <dlib/matrix.h>
#include <dlib/geometry.h>
#include <dlib/pixel.h>
#include <dlib/array2d/array2d_kernel.h>

class CERT
{
public:
	CERT();
	CERT(std::vector<dlib::shape_predictor>, double, std::vector<std::pair<dlib::full_object_detection, double>>, dlib::matrix<float, 0, 1>);
	~CERT();
	void serialize(std::ostream& out);
	void deserialize(std::istream& in);
	void setSubDivisions(std::vector<dlib::shape_predictor>);
	std::vector<dlib::shape_predictor> getSubDivisions();
	void setWeightedBins(std::vector<std::pair<dlib::full_object_detection, double>>);
	std::vector<std::pair<dlib::full_object_detection, double>>  getWeightedBins();
	void setInitShape(dlib::matrix<float, 0, 1>);
	dlib::matrix<float, 0, 1> getInitShape();
	dlib::full_object_detection PredictFinalShape(dlib::array2d<unsigned char> &, dlib::rectangle); // Gives full_object_detection when provided with an image and a detection/target area (from either object detector or testing files)
private:
	std::vector <dlib::shape_predictor> SubDivisions; // Vector for all loaded ERT models
	double TotalDeviation; // 
	std::vector < std::pair < dlib::full_object_detection, double >> WeightedBins;
	dlib::matrix<float, 0, 1> InitShape; // The InitShape used by all SubDivsions 

};

