Code can be found in a public Github repository link below.

https://github.com/jorgecfelix/OMSCS_MachineLearning

In the repository, the SupervisedLearning directory contains all the necessary code
for this first assignment.

For this assignment I wrote indvidual python scripts to run each of the algorithms:

    - decisiontree.py
    - adaboostedtree.py
    - knn.py
    - svm.py
    - neuralnet.py

Python 3.7.0 was used for this assignment with the packages and versions below:
    - pandas==0.24.2
    - numpy==1.16.0
    - scikit-learn==0.23.2
    - matplotlib==3.3.1
    - tensorflow==2.3.0



    
Datasets needed:
The two datasets used can be found through the links below.

Phishing Website Classification:
    https://archive.ics.uci.edu/ml/datasets/Phishing+Websites


For the first dataset (Phising Website) click on the link and then click on the data folder link provided.
Download the Training Dataset.arff file containing the data. Open the file and remove the attribute data at the 
top of the file and save. This file should be ready to use in one of the scripts above.

For example remove everything below from that file, and only leave the csv row data.

    @relation phishing
    
    @attribute having_IP_Address  { -1,1 }
    @attribute URL_Length   { 1,0,-1 }
    @attribute Shortining_Service { 1,-1 }
    @attribute having_At_Symbol   { 1,-1 }
    @attribute double_slash_redirecting { -1,1 }
    @attribute Prefix_Suffix  { -1,1 }
    @attribute having_Sub_Domain  { -1,0,1 }
    @attribute SSLfinal_State  { -1,1,0 }
    @attribute Domain_registeration_length { -1,1 }
    @attribute Favicon { 1,-1 }
    @attribute port { 1,-1 }
    @attribute HTTPS_token { -1,1 }
    @attribute Request_URL  { 1,-1 }
    @attribute URL_of_Anchor { -1,0,1 }
    @attribute Links_in_tags { 1,-1,0 }
    @attribute SFH  { -1,1,0 }
    @attribute Submitting_to_email { -1,1 }
    @attribute Abnormal_URL { -1,1 }
    @attribute Redirect  { 0,1 }
    @attribute on_mouseover  { 1,-1 }
    @attribute RightClick  { 1,-1 }
    @attribute popUpWidnow  { 1,-1 }
    @attribute Iframe { 1,-1 }
    @attribute age_of_domain  { -1,1 }
    @attribute DNSRecord   { -1,1 }
    @attribute web_traffic  { -1,0,1 }
    @attribute Page_Rank { -1,1 }
    @attribute Google_Index { 1,-1 }
    @attribute Links_pointing_to_page { 1,0,-1 }
    @attribute Statistical_report { -1,1 }
    @attribute Result  { -1,1 }
    
    
    @data

Census Data:
    https://archive.ics.uci.edu/ml/datasets/Census+Income

For the Census Data click on link above and then on the download folder.

Click on the adult.data file to download, this file should be ready to use with the scripts above.


How to run:

Each script takes in two arguments when running 

The first argument must be either d1 for dataset 1 which is the phishing dataset or d2 for the census dataset.


The second argument is the file corresponding with the dataset.

The scripts will then run and print their validation and learning plots as png files.


Examples for running a script with the Phishing website dataset:

    python decisiontree.py d1 path_to_file/Training_Dataset.arff

    python knn.py d1 path_to_file/Training_Dataset.arff

Examples for running a script with the Census dataset:

    python decisiontree.py d2 path_to_file/adult.data

    python knn.py d2 path_to_file/adult.data