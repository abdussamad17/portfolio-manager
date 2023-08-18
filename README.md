# portfolio-manager
The goal of this project is to build & compare portfolio construction and management using machine learning techniques. We compare the performance of the
models across several error metrics. A total of 17 model variants were tested from 8 classes of models.

## How to run
1. Install the regurements in the Requriements.txt file
2. Either run the script file which runs all the models using "sh run.sh" or run the exact model you would like to run using "python Testback.py <Nameofstrategy>,<parametername>=<parameter>"
3. The details of all the parameters can be found in the Testback and respective class file for the method you are trying to call

## Things to Note!
1. You need an API key from the data provider Financial Modelling Prep which should be put in an .env file
2. Youll need AWS S3 credentials to be able to upload results to the cloud
3. Youll need a Gurobi Optimiser licence installed on you local system. I used a free student licence which i got from their website
4. I have paramtereised the models that need GPU's to work on CPU's however it will be exxtremely slow.
