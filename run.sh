conda activate tensorml
cd /home/user/github/portfolio-manager


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

# Generate backtests
echo "Running backtests"
## - EqualDollarStrategy
echo "Running Testback.py EqualDollarStrategy"
python BackTest/Testback.py EqualDollarStrategy

## - EqualVolStrategy
echo "Running Testback.py EqualVolStrategy"
python BackTest/Testback.py EqualVolStrategy


## - MinimumVarianceStrategy
echo "Running Testback.py MinimumVarianceStrategy"
python BackTest/Testback.py MinimumVarianceStrategy


## - EqualVolContributionStrategy
echo "Running Testback.py EqualVolContributionStrategy"
python BackTest/Testback.py EqualVolContributionStrategy

## - MarkowitzStrategy
echo "Running Testback.py MarkowitzStrategy"

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=1"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=1

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=1"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=1

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=0.5,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=0.5,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=2,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=2,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.00017,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.00017,vol_weighted=False,max_concentration=0.05

echo "Running Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.00037,vol_weighted=False,max_concentration=0.05"
python BackTest/Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.00037,vol_weighted=False,max_concentration=0.05


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


## - HRPStrategy
echo "Running Testback.py XGBStrategy"
python BackTest/Testback.py XGBStrategy


## - CNNStrategy
echo "Running CNNStrategy"


### - CNNStrategy, strategy_type=equalpercent
echo "Running Testback.py CNNStrategy,strategy_type=equalpercent"
python BackTest/Testback.py CNNStrategy,strategy_type=equalpercent

### - CNNStrategy, strategy_type=marketindicator
echo "Running Testback.py CNNStrategy,strategy_type=marketindicator"
python BackTest/Testback.py CNNStrategy,strategy_type=marketindicator

### - CNNStrategy, strategy_type=sigmoid
echo "Running Testback.py CNNStrategy,strategy_type=sigmoid"
python BackTest/Testback.py CNNStrategy,strategy_type=sigmoid


# - Send Email
python run_notification.py