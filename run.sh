conda activate tensorml
cd /home/user/github/portfolio-manager

python run_notification.py "The script has begun"
# Download all the files
echo "Running data_fetch.py"
python DataAcquisition/data_fetch.py

# Generate extracts
echo "Generate extracts"
## - UniverseConstructor
echo "Running UniverseConstructor.py"
python DataAcquisition/UniverseConstructor.py
## - ExtractDailyPrice
echo "Running ExtractDailyPrice.py"
python DataAcquisition/ExtractDailyPrice.py

python run_notification.py "Data Extraction Complete"

cd /home/user/github/portfolio-manager/BackTest

# Generate backtests
echo "Running backtests"
## - EqualDollarStrategy
echo "Running Testback.py EqualDollarStrategy"
python BackTest/Testback.py EqualDollarStrategy

## - EqualVolStrategy
echo "Running Testback.py EqualVolStrategy"
python BackTest/Testback.py EqualVolStrategy

python ../run_notification.py "Equal dollar and equal vol complete"

## - MinimumVarianceStrategy
echo "Running Testback.py MinimumVarianceStrategy"
python BackTest/Testback.py MinimumVarianceStrategy


## - EqualVolContributionStrategy
echo "Running Testback.py EqualVolContributionStrategy"
python BackTest/Testback.py EqualVolContributionStrategy

python ../run_notification.py "minimum variance and Equal vol Contribution - complete"


## - HRPStrategy
echo "Running Testback.py HRPStrategy"

### - HRPStrategy,linkage_method=average
echo "Running Testback.py HRPStrategy,linkage_method=average"
python BackTest/Testback.py HRPStrategy,linkage_method=average

### - HRPStrategy,linkage_method=ward
echo "Running Testback.py HRPStrategy,linkage_method=ward"
python BackTest/Testback.py HRPStrategy,linkage_method=ward

### - HRPStrategy,linkage_method=single
echo "Running Testback.py HRPStrategy,linkage_method=single"
python BackTest/Testback.py HRPStrategy,linkage_method=single

python ../run_notification.py "starting XGBStrategy"
## - XGBStrategy-regression=False
echo "Running Testback.py XGBStrategy,regression=False"
python BackTest/Testback.py XGBStrategy,regression=False

## - XGBStrategy-regression=True
echo "Running Testback.py XGBStrategy,regression=True"
python BackTest/Testback.py XGBStrategy,regression=True

## - CNNStrategy
echo "Running CNNStrategy"

python ../run_notification.py "starting CNNStrategy"
### - CNNStrategy, strategy_type=equalpercent
echo "Running Testback.py CNNStrategy,strategy_type=equalpercent"
python BackTest/Testback.py CNNStrategy,strategy_type=equalpercent

### - CNNStrategy, strategy_type=marketindicator
echo "Running Testback.py CNNStrategy,strategy_type=marketindicator"
python BackTest/Testback.py CNNStrategy,strategy_type=marketindicator

### - CNNStrategy, strategy_type=sigmoid
echo "Running Testback.py CNNStrategy,strategy_type=sigmoid"
python BackTest/Testback.py CNNStrategy,strategy_type=sigmoid


python ../run_notification.py "starting Markowitz"

## - MarkowitzStrategy
echo "Running Testback.py MarkowitzStrategy"

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=1"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=1

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=1"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=1

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=0.05


# - Send Email
python ../run_notification.py "starting Markowitz"