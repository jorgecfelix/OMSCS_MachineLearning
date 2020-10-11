Code can be found in a public Github repository link below.

https://github.com/jorgecfelix/OMSCS_MachineLearning

In the repository, the RandomizedOptmization directory contains all the necessary code
for this first assignment.

For this assignment the scripts used to recreate the plots and experiments in the analysis are:

    - optimizations.py
    - neuralnet_a1.py
    - neuralnet_a2.py

Python 3.7.0 was used for this assignment with the packages and versions below:
    - pandas==0.24.2
    - numpy==1.16.0
    - scikit-learn==0.23.2
    - matplotlib==3.3.1
    - tensorflow==2.3.0
    - mlrose-hiive==2.1.3



    
Datasets needed:
The one dataset used can be found through the links below.

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

How to run:

To get the plots for the three problems youll need to run the optimizations.py file

The script needs an argument when ran, either "curves" or "tune"

When curves is provided it will generate 
the iterations vs fitness, problem size vs fitness, and problem size vs iterations
for each of the problems

When tune is provided it will generate the plots for each optimization algorithm and the hyper parameter tuned.


Examples for running the optimizations.py script with both values.

    python optimizations.py curves

    python optimizations.py tune


For the neural network there are two files 

    neuralnet_a1.py ( refers to tensorflow implementation in assignment one)
    
    neuralnet_a2.py ( refers to neural network created with mlrose and used with optimization algorithms)

Examples for running a script with the Phishing website dataset:

    python neuralnet_a1.py d1 path_to_file/Training_Dataset.arff

    python neuralnet_a2.py d1 path_to_file/Training_Dataset.arff

Both files will generate their respective plots.