#include <memory>
#include <sstream>
#include <string>
#include <dlib/gui_widgets.h>
#include <dlib/directed_graph.h>
#include <dlib/string.h>
#include <dlib/bayes_utils.h>
#include <dlib/set.h>
#include <dlib/graph_utils.h>
#include <dlib/stl_checked.h>


using namespace std;
using namespace dlib;

class win : public drawable_window
{
public:
	win(bool* StartTA, int* TrainSVMCTA, int* AggregateSVMCTA, string* SVMModelsInPTA, string* SVMModelOutPTA, int* TrainERTCTA, int* AggregateERTCTA, string* ERTModelsInPTA, string* ERTModelOutPTA, string* SVMTrainingDatasetPTA, string* SVMTestDatasetPTA, string* ERTTrainingDatasetPTA, string* ERTTestDatasetPTA
	) : // All widgets take their parent window as an argument to their constructor.
		Start(*this),
		StartTxT(*this),
		SelectSVMTrainingConfig(*this),
		SVMTrainConfigTxT(*this),
		SelectERTTrainingConfig(*this),
		ERTTrainConfigTxT(*this),
		SelectSVMAggregationConfig(*this),
		SVMAggregationConfigTxT(*this),
		SelectERTAggregationConfig(*this),
		ERTAggregationConfigTxT(*this),
		SelectSVMModelsInPath(*this),
		SVMModelsInPathTxT(*this),
		SelectSVMModelOutPath(*this),
		SVMModelOutPathTxT(*this),
		SelectERTModelsInPath(*this),
		ERTModelsInPathTxT(*this),
		SelectERTModelOutPath(*this),
		ERTModelOutPathTxT(*this),
		SelectSVMTrainingDatasetPath(*this),
		SVMTrainingDatasetPathTxT(*this),
		SelectSVMTestingDatasetPath(*this),
		SVMTestingDatasetPathTxT(*this),
		SelectERTTrainingDatasetPath(*this),
		ERTTrainingDatasetPathTxT(*this),
		SelectERTTestingDatasetPath(*this),
		ERTTestingDatasetPathTxT(*this),
		mbar(*this),
		StartS(StartTA),
		TrainSVMC(TrainSVMCTA),
		AggregateSVMC(AggregateSVMCTA),
		SVMModelsInP(SVMModelsInPTA),
		SVMModelOutP(SVMModelOutPTA),
		TrainERTC(TrainERTCTA),
		AggregateERTC(AggregateERTCTA),
		ERTModelsInP(ERTModelsInPTA),
		ERTModelOutP(ERTModelOutPTA),
		SVMTrainingDatasetP(SVMTrainingDatasetPTA),
		SVMTestDatasetP(SVMTestDatasetPTA),
		ERTTrainingDatasetP(ERTTrainingDatasetPTA),
		ERTTestDatasetP(ERTTestDatasetPTA)
	{
		// Creation of buttons 
		Start.set_pos(10, 60);
		Start.set_name("Start program, set other settings first");
		SelectSVMTrainingConfig.set_pos(10, 120);
		SelectSVMTrainingConfig.set_name("Enable/disable SVM model training and testing for object detection, model shall be agregated if aggregation is set");
		SelectERTTrainingConfig.set_pos(10, 180);
		SelectERTTrainingConfig.set_name("Enable/disable ERT model training and testing for landmark placement, model shall be agregated if aggregation is set");
		SelectSVMAggregationConfig.set_pos(10, 240);
		SelectSVMAggregationConfig.set_name("Select SVM model aggregation/test mode for object detection: 0 = Do nothing, 1 = Aggregate input (and trained model), 2 = Test selected input models only");
		SelectERTAggregationConfig.set_pos(10, 300);
		SelectERTAggregationConfig.set_name("Select ERT model aggregation/test mode for landmark detection: 0 = Do nothing, 1 = Aggregate input (and trained model), 2 = Test selected input ERT models only, 3 = Test CERT models only");
		SelectSVMModelsInPath.set_pos(10, 360);
		SelectSVMModelsInPath.set_name("Select input path for SVM model(s), use *.svm to select all SVM models in a folder");
		SelectSVMModelOutPath.set_pos(10, 420);
		SelectSVMModelOutPath.set_name("Select output path for new SVM model");
		SelectERTModelsInPath.set_pos(10, 480);
		SelectERTModelsInPath.set_name("Select input path for ERT model(s), use *.dat to select all ERT models in a folder, or *CERT for all CERT models in case of testing only mode");
		SelectERTModelOutPath.set_pos(10, 540);
		SelectERTModelOutPath.set_name("Select output path for new ERT model (.dat) or CERT model (.CERT)");
		SelectSVMTrainingDatasetPath.set_pos(10, 600);
		SelectSVMTrainingDatasetPath.set_name("Select path for SVM model training dataset .xml file");
		SelectSVMTestingDatasetPath.set_pos(10, 660);
		SelectSVMTestingDatasetPath.set_name("Select path for SVM model testing dataset .xml file");
		SelectERTTrainingDatasetPath.set_pos(10, 720);
		SelectERTTrainingDatasetPath.set_name("Select path for ERT training dataset .xml file");
		SelectERTTestingDatasetPath.set_pos(10, 780);
		SelectERTTestingDatasetPath.set_name("Select path for ERT testing dataset .xml file");
	
		// Creation of labels
		StartTxT.set_pos(Start.left(), Start.bottom() + 5);
		SVMTrainConfigTxT.set_pos(SelectSVMTrainingConfig.left(), SelectSVMTrainingConfig.bottom() + 5);
		ERTTrainConfigTxT.set_pos(SelectERTTrainingConfig.left(), SelectERTTrainingConfig.bottom() + 5);
		SVMAggregationConfigTxT.set_pos(SelectSVMAggregationConfig.left(), SelectSVMAggregationConfig.bottom() + 5);
		ERTAggregationConfigTxT.set_pos(SelectERTAggregationConfig.left(), SelectERTAggregationConfig.bottom() + 5);
		SVMModelsInPathTxT.set_pos(SelectSVMModelsInPath.left(), SelectSVMModelsInPath.bottom() + 5);
		SVMModelOutPathTxT.set_pos(SelectSVMModelOutPath.left(), SelectSVMModelOutPath.bottom() + 5);
		ERTModelsInPathTxT.set_pos(SelectERTModelsInPath.left(), SelectERTModelsInPath.bottom() + 5);
		ERTModelOutPathTxT.set_pos(SelectERTModelOutPath.left(), SelectERTModelOutPath.bottom() + 5);
		SVMTrainingDatasetPathTxT.set_pos(SelectSVMTrainingDatasetPath.left(), SelectSVMTrainingDatasetPath.bottom() + 5);
		SVMTestingDatasetPathTxT.set_pos(SelectSVMTestingDatasetPath.left(), SelectSVMTestingDatasetPath.bottom() + 5);
		ERTTrainingDatasetPathTxT.set_pos(SelectERTTrainingDatasetPath.left(), SelectERTTrainingDatasetPath.bottom() + 5);
		ERTTestingDatasetPathTxT.set_pos(SelectERTTestingDatasetPath.left(), SelectERTTestingDatasetPath.bottom() + 5);
		
		// Link buttons to function events
		Start.set_click_handler(*this, &win::on_Start_clicked);
		SelectSVMTrainingConfig.set_click_handler(*this, &win::on_SelectSVMTrainingConfig_clicked);
		SelectERTTrainingConfig.set_click_handler(*this, &win::on_SelectERTTrainingConfig_clicked);
		SelectSVMAggregationConfig.set_click_handler(*this, &win::on_SelectSVMAggregationConfig_clicked);
		SelectERTAggregationConfig.set_click_handler(*this, &win::on_SelectERTAggregationConfig_clicked);
		SelectSVMModelsInPath.set_click_handler(*this, &win::on_SelectSVMModelsInPath_clicked);
		SelectSVMModelOutPath.set_click_handler(*this, &win::on_SelectSVMModelOutPath_clicked);
		SelectERTModelsInPath.set_click_handler(*this, &win::on_SelectERTModelsInPath_clicked);
		SelectERTModelOutPath.set_click_handler(*this, &win::on_SelectERTModelOutPath_clicked);
		SelectSVMTrainingDatasetPath.set_click_handler(*this, &win::on_SelectSVMTrainingDatasetPath_clicked);
		SelectSVMTestingDatasetPath.set_click_handler(*this, &win::on_SelectSVMTestingDatasetPath_clicked);
		SelectERTTrainingDatasetPath.set_click_handler(*this, &win::on_SelectERTTrainingDatasetPath_clicked);
		SelectERTTestingDatasetPath.set_click_handler(*this, &win::on_SelectERTTestingDatasetPath_clicked);

		// Creation of simple menu bar
		mbar.set_number_of_menus(1);
		// Adding shortcut to menu of alt+M
		mbar.set_menu_name(0, "Menu", 'M');
		// Adding a separator (i.e. a horizontal separating line) to the menu
		mbar.menu(0).add_menu_item(menu_item_separator());
		// Adding about window
		mbar.menu(0).add_menu_item(menu_item_text("About", *this, &win::show_about, 'A'));

		set_size(700, 600);
		set_title("EyeBlink program: Model training and aggregation V0.4");
		show();

		// Initional text display of labels
		ostringstream sout;
		sout.str("Program is ready to start once all settings are done.");
		StartTxT.set_text(sout.str());
		sout.str("SVM model training is now disabled");
		SVMTrainConfigTxT.set_text(sout.str());
		sout.str("ERT model training is now disabled");
		ERTTrainConfigTxT.set_text(sout.str());
		sout.str("Current SVM model aggregation mode:" + to_string(*AggregateSVMC));
		SVMAggregationConfigTxT.set_text(sout.str());
		sout.str("Current ERT model aggregation mode:" + to_string(*AggregateERTC));
		ERTAggregationConfigTxT.set_text(sout.str());
		sout.str(*SVMModelsInP);
		SVMModelsInPathTxT.set_text(sout.str());
		sout.str(*SVMModelOutP);
		SVMModelOutPathTxT.set_text(sout.str());
		sout.str(*ERTModelsInP);
		ERTModelsInPathTxT.set_text(sout.str());
		sout.str(*ERTModelOutP);
		ERTModelOutPathTxT.set_text(sout.str());
		sout.str(*SVMTrainingDatasetP);
		SVMTrainingDatasetPathTxT.set_text(sout.str());
		sout.str(*SVMTestDatasetP);
		SVMTestingDatasetPathTxT.set_text(sout.str());
		sout.str(*ERTTrainingDatasetP);
		ERTTrainingDatasetPathTxT.set_text(sout.str());
		sout.str(*ERTTestDatasetP);
		ERTTestingDatasetPathTxT.set_text(sout.str());
	}

	~win(
	)
	{
		// Closing of window during deconstruction
		close_window();
	}

private:

	void on_Start_clicked(
	)
	{
		ostringstream sout;
		sout << "Program has started, setting changes should no longer be made and could lead to severe problems.";
		StartTxT.set_text(sout.str());
		*StartS = 1;
	}

	void on_SelectSVMTrainingConfig_clicked(
	)
	{
		ostringstream sout;
		if (*TrainSVMC == 0)
		{
			sout.str("SVM model training is now enabled, a special training settings window shall show during runtime");
			SVMTrainConfigTxT.set_text(sout.str());
			*TrainSVMC = 1;
		}
		else
		{
			sout.str("SVM model training is now disabled");
			SVMTrainConfigTxT.set_text(sout.str());
			*TrainSVMC = 0;
		}
	}

	void on_SelectERTTrainingConfig_clicked(
	)
	{
		ostringstream sout;
		if (*TrainERTC == 0)
		{
			sout.str("ERT model training is now enabled, a special training settings window shall show during runtime");
			ERTTrainConfigTxT.set_text(sout.str());
			*TrainERTC = 1;
		}
		else
		{
			sout.str("ERT model training is now disabled");
			ERTTrainConfigTxT.set_text(sout.str());
			*TrainERTC = 0;
		}
	}

	void on_SelectSVMAggregationConfig_clicked(
	)
	{
		if (*AggregateSVMC == 2)
			*AggregateSVMC = 0;
		else if (*AggregateSVMC == 1)
			*AggregateSVMC = 2;
		else
			*AggregateSVMC = 1;
		ostringstream sout;
		sout << "Current SVM model aggregation mode:" + to_string(*AggregateSVMC);
		SVMAggregationConfigTxT.set_text(sout.str());
	}

	void on_SelectERTAggregationConfig_clicked(
	)
	{
		if (*AggregateERTC == 0)
			*AggregateERTC = 1;
		else if (*AggregateERTC == 1)
			*AggregateERTC = 2;
		else if (*AggregateERTC == 2)
			*AggregateERTC = 3;
		else
			*AggregateERTC = 0;
		ostringstream sout;
		sout << "Current ERT model aggregation mode:" + to_string(*AggregateERTC);
		ERTAggregationConfigTxT.set_text(sout.str());
	}

	void on_SelectSVMModelsInPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectSVMModelsInPath_selected);
	}

	void on_SelectSVMModelOutPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectSVMModelOutPath_selected);
	}

	void on_SelectERTModelsInPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectERTModelsInPath_selected);
	}

	void on_SelectERTModelOutPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectERTModelOutPath_selected);
	}

	void on_SelectSVMTrainingDatasetPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectSVMTrainingDatasetPath_selected);
	}

	void on_SelectSVMTestingDatasetPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectSVMTestingDatasetPath_selected);
	}

	void on_SelectERTTrainingDatasetPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectERTTrainingDatasetPath_selected);
	}

	void on_SelectERTTestingDatasetPath_clicked(
	)
	{
		open_file_box(*this, &win::on_SelectERTTestingDatasetPath_selected);
	}

	void show_about(
	)
	{
		message_box("About", "This is the EyeBlink program: Model training and aggregation subsystem. Created by Stefan Zwaard, based on the work of Paul Baker. Version 0.5, 16-06-2020");
	}

	void on_SelectSVMModelsInPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*SVMModelsInP = file_name;
		SVMModelsInPathTxT.set_text(sout.str());
	}

	void on_SelectSVMModelOutPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*SVMModelOutP = file_name;
		SVMModelOutPathTxT.set_text(sout.str());
	}

	void on_SelectERTModelsInPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*ERTModelsInP = file_name;
		ERTModelsInPathTxT.set_text(sout.str());
	}

	void on_SelectERTModelOutPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*ERTModelOutP = file_name;
		ERTModelOutPathTxT.set_text(sout.str());
	}

	void on_SelectSVMTrainingDatasetPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*SVMTrainingDatasetP = file_name;
		SVMTrainingDatasetPathTxT.set_text(sout.str());
	}

	void on_SelectSVMTestingDatasetPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*SVMTestDatasetP = file_name;
		SVMTestingDatasetPathTxT.set_text(sout.str());
	}

	void on_SelectERTTrainingDatasetPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*ERTTrainingDatasetP = file_name;
		ERTTrainingDatasetPathTxT.set_text(sout.str());
	}

	void on_SelectERTTestingDatasetPath_selected(const std::string& file_name)
	{
		ostringstream sout;
		sout << file_name;
		*ERTTestDatasetP = file_name;
		ERTTestingDatasetPathTxT.set_text(sout.str());
	}

	button Start;
	label StartTxT;
	button SelectSVMTrainingConfig;
	label SVMTrainConfigTxT;
	button SelectERTTrainingConfig;
	label ERTTrainConfigTxT;
	button SelectSVMAggregationConfig;
	label SVMAggregationConfigTxT;
	button SelectERTAggregationConfig;
	label ERTAggregationConfigTxT;
	button SelectSVMModelsInPath;
	label SVMModelsInPathTxT;
	button SelectSVMModelOutPath;
	label SVMModelOutPathTxT;
	button SelectERTModelsInPath;
	label ERTModelsInPathTxT;
	button SelectERTModelOutPath;
	label ERTModelOutPathTxT;
	button SelectSVMTrainingDatasetPath;
	label SVMTrainingDatasetPathTxT;
	button SelectSVMTestingDatasetPath;
	label SVMTestingDatasetPathTxT;
	button SelectERTTrainingDatasetPath;
	label ERTTrainingDatasetPathTxT;
	button SelectERTTestingDatasetPath;
	label ERTTestingDatasetPathTxT;
	menu_bar mbar;
	bool* StartS;
	int* TrainSVMC;
	int* AggregateSVMC;
	string* SVMModelsInP;
	string* SVMModelOutP;
	int* TrainERTC;
	int* AggregateERTC;
	string* ERTModelsInP;
	string* ERTModelOutP;
	string* SVMTrainingDatasetP;
	string* SVMTestDatasetP;
	string* ERTTrainingDatasetP;
	string* ERTTestDatasetP;
};

// Window for SVM model training settings, only shown during runtime if SVM training is selected
class SVMsettingsWin : public drawable_window
{
public:
	SVMsettingsWin(bool* DoneTA, int* ThreadsTA, double* CTA, double* epsTA, unsigned int* num_foldTA, unsigned long* target_SizeTA, unsigned long* upsample_amountTA
	) :
		Done(*this),
		ThreadField(*this),
		CField(*this),
		epsField(*this),
		num_foldField(*this),
		target_SizeField(*this),
		upsample_amountField(*this),
		ThreadFieldName(*this),
		CFieldName(*this),
		epsFieldName(*this),
		num_foldFieldName(*this),
		target_SizeFieldName(*this),
		upsample_amountFieldName(*this),
		DoneS(DoneTA),
		ThreadsL(ThreadsTA),
		CL(CTA),
		epsL(epsTA),
		num_foldL(num_foldTA),
		target_SizeL(target_SizeTA),
		upsample_amountL(upsample_amountTA)
	{
		
		// Giving all buttons, labels and text labels a position and text where needed
		Done.set_pos(10, 60);
		Done.set_name("If all SVM model training settings are correct, press this to continue");
		ThreadField.set_pos(10, 120);
		ThreadField.set_width(50);
		ThreadField.set_text(to_string(*ThreadsL));
		ThreadFieldName.set_pos(10, 105);
		ThreadFieldName.set_text("Number of threads usable by system: Use atleast 1, current system cores -2 advised. Default = 4");
		CField.set_pos(10, 180);
		CField.set_width(50);
		CField.set_text(to_string(*CL));
		CFieldName.set_pos(10, 165);
		CFieldName.set_text("C. Default = 1.0");
		epsField.set_pos(10, 240);
		epsField.set_width(50);
		epsField.set_text(to_string(*epsL));
		epsFieldName.set_pos(10, 225);
		epsFieldName.set_text("eps. Default = 0.01");
		num_foldField.set_pos(10, 300);
		num_foldField.set_width(50);
		num_foldField.set_text(to_string(*num_foldL));
		num_foldFieldName.set_pos(10, 285);
		num_foldFieldName.set_text("num_fold. Default = 3");
		target_SizeField.set_pos(10, 360);
		target_SizeField.set_width(50);
		target_SizeField.set_text(to_string(*target_SizeL));
		target_SizeFieldName.set_pos(10, 345);
		target_SizeFieldName.set_text("target_size, product is taken from entry. Default = 80 * 80");
		upsample_amountField.set_pos(10, 420);
		upsample_amountField.set_width(50);
		upsample_amountField.set_text(to_string(*upsample_amountL));
		upsample_amountFieldName.set_pos(10, 405);
		upsample_amountFieldName.set_text("upsample_amount. Default = 0");

		
		// Linking button and function
		Done.set_click_handler(*this, &SVMsettingsWin::on_Done_clicked);
		
		// General window settings
		set_size(700, 500);
		set_title("SVM model training settings window");
		show();
	}
	~SVMsettingsWin(
	)
	{
		// Closing of window during deconstruction
		close_window();
	}
private:
	void on_Done_clicked(
	)
	{
		*DoneS = 1;
		// Reading all text fields
		*ThreadsL = stoi(ThreadField.text());
		*CL = stod(CField.text());
		*epsL = stod(epsField.text());
		*num_foldL = stoul(num_foldField.text());
		*target_SizeL = stoul(target_SizeField.text());
		*upsample_amountL = stoul(upsample_amountField.text());
	}

	button Done;
	text_field ThreadField;
	label ThreadFieldName;
	text_field CField;
	label CFieldName;
	text_field epsField;
	label epsFieldName;
	text_field num_foldField;
	label num_foldFieldName;
	text_field target_SizeField;
	label target_SizeFieldName;
	text_field upsample_amountField;
	label upsample_amountFieldName;
	bool* DoneS;
	int* ThreadsL;
	double* CL;
	double* epsL;
	unsigned int* num_foldL;
	unsigned long* target_SizeL;
	unsigned long* upsample_amountL;
};




// Window for ERT model training settings, only shown during runtime if ERT training is selected
class ERTsettingsWin : public drawable_window
{
public:
	ERTsettingsWin(bool* DoneTA, unsigned int* ThreadsTA, unsigned long* oversampling_amountTA, double* nuTA, unsigned long* tree_depthTA, unsigned int* feature_pool_sizeTA, unsigned int* num_test_splitsTA, unsigned int* cascade_depthTA, unsigned long* num_trees_per_cascade_levelTA, double* lambdaTA
	) :
		ERTDone(*this),
		ERTThreadField(*this),
		ERTThreadFieldName(*this),
		oversampling_amountField(*this),
		oversampling_amountFieldName(*this),
		nuField(*this),
		nuFieldName(*this),
		tree_depthField(*this),
		tree_depthFieldName(*this),
		feature_pool_sizeField(*this),
		feature_pool_sizeFieldName(*this),
		num_test_splitsField(*this),
		num_test_splitsFieldName(*this),
		cascade_depthField(*this),
		cascade_depthFieldName(*this),
		num_trees_per_cascade_levelField(*this),
		num_trees_per_cascade_levelFieldName(*this),
		lambdaField(*this),
		lambdahFieldName(*this),
		DoneS(DoneTA),
		ThreadsL(ThreadsTA),
		oversampling_amountL(oversampling_amountTA),
		nuL(nuTA),
		tree_depthL(tree_depthTA),
		feature_pool_sizeL(feature_pool_sizeTA),
		num_test_splitsL(num_test_splitsTA),
		cascade_depthL(cascade_depthTA),
		num_trees_per_cascade_levelL(num_trees_per_cascade_levelTA),
		lambdaL(lambdaTA)
	{
		// Giving all buttons, labels and text labels a position and text where needed
		ERTDone.set_pos(10, 60);
		ERTDone.set_name("If all ERT model training settings are correct, press this to continue");
		ERTThreadField.set_pos(10, 120);
		ERTThreadField.set_width(50);
		ERTThreadField.set_text(to_string(*ThreadsL));
		ERTThreadFieldName.set_pos(10, 105);
		ERTThreadFieldName.set_text("Number of threads usable by system: Use atleast 1, current system cores -2 advised. Default = 4");
		oversampling_amountField.set_pos(10, 180);
		oversampling_amountField.set_width(50);
		oversampling_amountField.set_text(to_string(*oversampling_amountL));
		oversampling_amountFieldName.set_pos(10, 165);
		oversampling_amountFieldName.set_text("oversampling_amount. Default is 0, or 50 for smaller datasets ");
		nuField.set_pos(10, 240);
		nuField.set_width(50);
		nuField.set_text(to_string(*nuL));
		nuFieldName.set_pos(10, 225);
		nuFieldName.set_text("nu. Default is 0.1 for small model capicity ");
		tree_depthField.set_pos(10, 300);
		tree_depthField.set_width(50);
		tree_depthField.set_text(to_string(*tree_depthL));
		tree_depthFieldName.set_pos(10, 285);
		tree_depthFieldName.set_text("tree_depth. Default is 5");
		feature_pool_sizeField.set_pos(10, 360);
		feature_pool_sizeField.set_width(50);
		feature_pool_sizeField.set_text(to_string(*feature_pool_sizeL));
		feature_pool_sizeFieldName.set_pos(10, 345);
		feature_pool_sizeFieldName.set_text("feature_pool_size. Default is 400 for small model capicity ");
		num_test_splitsField.set_pos(10, 420);
		num_test_splitsField.set_width(50);
		num_test_splitsField.set_text(to_string(*num_test_splitsL));
		num_test_splitsFieldName.set_pos(10, 405);
		num_test_splitsFieldName.set_text("num_test_splits. Default is 20");
		cascade_depthField.set_pos(10, 480);
		cascade_depthField.set_width(50);
		cascade_depthField.set_text(to_string(*cascade_depthL));
		cascade_depthFieldName.set_pos(10, 465);
		cascade_depthFieldName.set_text("Cascade_depth. Default is 10, also known as number of ERT iterations");
		num_trees_per_cascade_levelField.set_pos(10, 540);
		num_trees_per_cascade_levelField.set_width(50);
		num_trees_per_cascade_levelField.set_text(to_string(*num_trees_per_cascade_levelL));
		num_trees_per_cascade_levelFieldName.set_pos(10, 525);
		num_trees_per_cascade_levelFieldName.set_text("num_trees_per_cascade_level. Default is 500");
		lambdaField.set_pos(10, 600);
		lambdaField.set_width(50);
		lambdaField.set_text(to_string(*lambdaL));
		lambdahFieldName.set_pos(10, 585);
		lambdahFieldName.set_text("Lambda. Default is 0.1");

		
		// Linking button and function
		ERTDone.set_click_handler(*this, &ERTsettingsWin::on_ERTDone_clicked);

		// General window settings
		set_size(700, 500);
		set_title("ERT model training settings window");
		show();
	}
	~ERTsettingsWin(
	)
	{
		// Closing of window during deconstruction
		close_window();
	}

private:
	void on_ERTDone_clicked(
	)
	{
		*DoneS = 1;
		// Reading all text fields
		*ThreadsL = stoi(ERTThreadField.text());
		*oversampling_amountL = stod(oversampling_amountField.text());
		*nuL = stod(nuField.text());
		*tree_depthL = stoul(tree_depthField.text());
		*feature_pool_sizeL = stoi(feature_pool_sizeField.text());
		*num_test_splitsL = stoi(num_test_splitsField.text());
		*cascade_depthL = stoi(cascade_depthField.text());
		*num_trees_per_cascade_levelL = stoul(num_trees_per_cascade_levelField.text());
		*lambdaL = stod(lambdaField.text());
	}

	button ERTDone;
	text_field ERTThreadField;
	label ERTThreadFieldName;
	text_field oversampling_amountField;
	label oversampling_amountFieldName;
	text_field nuField;
	label nuFieldName;
	text_field tree_depthField;
	label tree_depthFieldName;
	text_field feature_pool_sizeField;
	label feature_pool_sizeFieldName;
	text_field num_test_splitsField;
	label num_test_splitsFieldName;
	text_field cascade_depthField;
	label cascade_depthFieldName;
	text_field num_trees_per_cascade_levelField;
	label num_trees_per_cascade_levelFieldName;
	text_field lambdaField;
	label lambdahFieldName;
	bool* DoneS;
	unsigned int* ThreadsL;
	unsigned long* oversampling_amountL;
	double* nuL;
	unsigned long* tree_depthL;
	unsigned int* feature_pool_sizeL;
	unsigned int* num_test_splitsL;
	unsigned int* cascade_depthL;
	unsigned long* num_trees_per_cascade_levelL;
	double* lambdaL; 
};
	
