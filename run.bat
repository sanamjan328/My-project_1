@ECHO OFF
REM Runs both my project scripts

python C:\local\My_demo\Example_3\1_prepareData.py
ECHO Ran 1_prepareData
python C:\local\My_demo\Example_3\2_refineDNN.py
ECHO Ran 2_refineDNN
python C:\local\My_demo\Example_3\3_featurizeImages.py
ECHO Ran 3_featurizeImages
python C:\local\My_demo\Example_3\4_trainSVM.py
ECHO Ran 4_trainSVM
python C:\local\My_demo\Example_3\5_evaluate.py
ECHO Ran 5_evaluate