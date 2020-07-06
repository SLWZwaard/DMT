# DMT
Distributed Model Training: Distributed training, testing and aggregation using WBA and MWMA for object detection and landmark localization models.
Created by S.L.W. Zwaard from the Hauge University of Applied Sciences.

# General introduction
This program is designed and created to train, aggregate and test models for object detection and landmark localization.
Using a GUI, the user can select to train new models, combine existing or newly trained models, or test existing models. Training and testing datasets can also be selected using the GUI.
The object detection models consist of a Support Vector Machine (SVM) classifier with a Histogram of Orientated Gradients (HOG) feature extractor, using the DLib library implementation.
For landmark localization, the Ensemble of Regression Tree (ERT) implementation of DLib is implemented. For both models, the default DLib implementation for training is situated inside the program and ran when configured by trough the GUI.
Training settings are also adjusted trough the GUI when the program is run with training mode enabled. If not all training data is present on the same node, for example because of privacy concerns, the system can make use of distributed training trough the aggregation of locally trained models.
For model aggregation, the Weighted Bin Aggregation (WBA) algorithm is used for the ERT models and the Mean Weighted Matrix Aggregation (MWMA) algorithm is used for the SVM with HOG models.
When combining a SVM model, a mean average is taken for the matrix of new model, representing the models used as input. The new SVM model can be used in the same manner as normal SVM models and require no changes to other DLib programs.
The combination of ERT models is done by creating a larger forest with weighted subresults, from which an average result is taken when the model is used. A combined ERT model uses the same code as a normal ERT model, but adds on a layer for the calculation and averaging of sub results to the final shape prediction.
Therefore, the combined ERT models are saved as CERT models. The CERT models are mostly the same in use and give the same return in the shape predict functions as normal ERT models, but the inclusion of the CERT code files are needed for a DLib project to make use of the new combined models.
For a detailed explanation of both algorithms, please see the paper given in the Reference section below.

# File descriptions
ModelTrainingAndAggregation.cpp is the main source code file an has the main function. It activates the training and aggregation process as well as IO saving and GUI activation.
It requires the Window.h and Cert.h files, as well as the DLib and OpenCV Library to compile.

The Window.h file is responsible for all GUI windows, it makes use of the DLib GUI functionality. Both the main GUI window, as well as the two smaller windows for the training settings configuration can be found here.
The windows are controlled from the main file, communication of data is done trough memory pointers. There is no Window.cpp file, inline implementation is used.

The CERT.h and CERT.cpp provide the Combined Ensamble of Regrassion Trees functionality for saving/loading and running the shape prediction. The CERT models are created in the main program, but the running code for shape prediction is done here.
These files are needed in other programs if a CERT model is used, while SVM models can be used with the DLib code alone. These files require the DLib library to work, as they only form an extra layer on top of the DLib code for shape prediction.

Please see the paper given in the Reference section below for more details on the creation process of both model types, as well as the running process of the CERT models. More details are also given trough commentary inside the code files themselves.

The Release\Distributed Model Training and Aggregation.exe file is the executable program which uses the opencv_world430.dll file. The release files have been given in case only the program is needed, the executable is rebuild when the build instructions below are followed.

# Build requirements
The given source code is dependent on the DLib and OpenCV libraries. The use of a Windows 10 64 bits based operating system might be required, although other Windows systems may work as well.
Ensure that the DLib source folder is added to the parent folder of the source code folder. So the parent folder has both the DLib source folder as well as the new source code.
Ensure CMake and OpenCV is installed, as well as a C++11 compiler. For the original build, Microsoft Visual Studio was used.

# Build instructions
Download the source code to a folder of choice. 
Change the OpenCV_DIR to the correct path in the CMakeLists.txt if needed.
From CMD:
-cd to current directory
cmake .
cmake --build . --config Release
This should, assuming that Microsoft Visual Studio is installed, create a Microsoft Visual Studio Project file, as well as compiling the code using the last command.
Please see the Using DLib from C++ section on http://dlib.net/compile.html  for more information.

Another option is to create s MS Visual Studio project file (or other IDE) using the CMake GUI program. The current code files can also be added to an existing DLib/OpenCV project.

Always use the AVX extended instructions on Release x64 for best performance of program

# Program requirements
The program is designed for a Windows 10 operating system, it may run on earlier Windows versions, but it's not supported. Neither are Linux or IOS based operating systems.
Ensure the opencv_world430.dll (or other OpenCV.dll if other version is used during build) is present in the same folder as the .exe file before executing the .exe fail. Failure to do so shall result in an error on program startup.
For training and testing object detection or landmark localization models, ensure training or testing datasets with the appropriate DLib .xml format for annotation is available. 
Also ensure read rights of any training or testing dataset and read/write rights for the folder where any new models are saved. Also ensure the program itself has appropriate rights, use of administrator rights might be needed for your situation.

# Program usage
The program can be started from the Windows GUI, or from the Windows CMD. Running the program from an elevated CMD has preference, as then exception or error messages do not disappear if the program is shut down abruptly. 
Please note that the GUI windows open too small for all buttons to be visible, stretch the window size to ensure all configurations are set before proceeding.
Information about what each button does is provided through the GUI. The program can be set to train either SVM or ERT models or both. Enabling training shall open an extra window later where training configuration can be provided.
Paths for model output and training and testing datasets must be given in the main window first. Model aggregation or testing can also be configured in the main window. The model resulting from aggregation is saved on the same path as for the training process. If 
both training and aggregation of a model type is selected, the trained model is used as input in the aggregation process. For model aggregation and testing, a test dataset must always be specified, as well as models to be used.
Multiple models can be selected trough the use of the asterisk character. For example, all SVM models in a folder can be used with the "\*.svm" path.
Once all settings are provided in the main window, the start program button on top can be used to proceed with the training, testing or aggregation process. Once the program is completed, new settings can be provided, and the start button can be used again for the next run.

# Reference
This content is developed as part of a joint research project for the Computer Engineering lab at Delft University of Technology, the Neuroscience department of the Erasmus Medical Center and the Babylab from the Princeton Neuroscience Institute. If any of this is of use to you, please include the following reference in your related work:

// Add ref to paper here.
