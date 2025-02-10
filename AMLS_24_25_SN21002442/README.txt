Organisation of the project:

- folder A contains task_A.py, which is the code for task A.
- folder B contains task_B.py, which is the code for task B.
- folder Datasets contsins two sub folders, which are BreastMNIST and BloodMNIST. 
   Data for task A and B should be stored in the following sub floder accordingly.
-main.py is the main python code, run the entire project here, instruction will
 be displayed after this file is executed 

Packages required to run :
- PyTorch 
-TorchVision
- MedMNIST
- Matplotlib
- NumPy

Please aware that you can choose to use the medminst api to download the split and labelled 
datasets or read data from the dataset folder and spit them into train/validate set, the test set 
is still obtained from the Medminst api.

if you decide to read data from the dataset folder, just simply comment out the dataloader function 
which utilises the medminst api and uncomment the other dataloader function in task_A.py and task_B.py.

remember to check and correct the location path of the data files from line 51 in task_A.py and 
line 49 in task_B.py

