To reproduce the results of this project, firstly  the dataset should be downloaded from  'https://www.kaggle.com/datasets/briscdataset/brisc2025'.  

To divide it into trining and validation, use the code file  named 'dataset_prep_train'. To run that file, pass the path of the train file whihc is found isnide the downloaded folder from the link above to the
variable named 'base_dir'


The above script will create 'data' and 'labels.cssv' file inside 'train' folder. pass the paths of those folders to the variables named 'train_imge_dir' abd 'val_csv' in the training script of each model. 
Similarly, the script will create a folder named 'validate' that contains 'data' and 'labels.csv' whose paths should be passed to the variables 'val_img_dir' abd 'val_csv'  respectively inside the trainign script. 

To train a model, run the script named by the name of that model (e.g. run the script 'vgg16' to train vgg16). 

To get the testing data, run the script named 'dataset_prep_test' and pass to the variable 'base_dir'  (in the preparation script).
The script will create a 'data folder' and 'labels.csv' whose paths should be passed to the variables 'test_img_dire' and 'test_csv' respectively
inside the testing script of each model. 

The test script of each model is named   test_name_of-the_model (for example the testing script for vgg16 is named 'test_vgg16')

After training all the eight models. the script named 'skripsi_3_models.py' can be used to test the soft voting ensemble. 


To evaluate models on the external data, download the external data first from the link 'https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset'.

Do the smme procedure on it as the testing set (use the 'dataset_prep_test.py' script but with the paths of the new datasets). 

To obtain the accuracy and the confusion matrix for a particular model on the external data, run testing script of that model but pass to it the variables 
of the external data in the 'test_img_dir' and the 'test_csv'
