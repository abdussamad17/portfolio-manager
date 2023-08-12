conda activate tensorml
cd /home/user/github/portfolio-manager

python run_notification.py "The script has begun"
## Download all the files
#echo "Running data_fetch.py"
#python DataAcquisition/data_fetch.py
#
## Generate extracts
#echo "Generate extracts"
### - UniverseConstructor
#echo "Running UniverseConstructor.py"
#python DataAcquisition/UniverseConstructor.py
### - ExtractDailyPrice
#echo "Running ExtractDailyPrice.py"
#python DataAcquisition/ExtractDailyPrice.py
#
#python run_notification.py "Data Extraction Complete"

cd /home/user/github/portfolio-manager/BackTest

# Generate backtests
#echo "Running backtests"
#
### - EqualDollarStrategy && EqualVolStrategy
#echo "Running Testback.py EqualDollarStrategy and EqualVolStrategy "
#python Testback.py EqualDollarStrategy &
#python Testback.py EqualVolStrategy &
#
#wait
#
#python ../run_notification.py "Equal dollar and equal vol complete"
#
### - MinimumVarianceStrategy && EqualVolContributionStrategy
#echo "Running Testback.py MinimumVarianceStrategy & EqualVolContributionStrategy"
#python Testback.py MinimumVarianceStrategy &
#python Testback.py EqualVolContributionStrategy &
#
#wait
#
#python ../run_notification.py "minimum variance and Equal vol Contribution - complete"
#
#
### - HRPStrategy
#echo "Running Testback.py HRPStrategies=average,ward,single"
#
#python Testback.py HRPStrategy,linkage_method=average &
#
#python Testback.py HRPStrategy,linkage_method=ward &
#
#python Testback.py HRPStrategy,linkage_method=single &
#
#wait
#
#python ../run_notification.py "starting XGBStrategy,regression=False and True"
### - XGBStrategy-regression=False,regression=True
#
#python Testback.py XGBStrategy,regression=False &
#python Testback.py XGBStrategy,regression=True&
#
#wait

## - CNNStrategy
echo "Running CNNStrategy"

python ../run_notification.py "starting CNNStrategy"
### - CNNStrategy, strategy_type=equalpercent
echo "Running Testback.py CNNStrategy,strategy_type=equalpercent"
python Testback.py CNNStrategy,strategy_type=equalpercent

### - CNNStrategy, strategy_type=marketindicator
echo "Running Testback.py CNNStrategy,strategy_type=marketindicator"
python Testback.py CNNStrategy,strategy_type=marketindicator

### - CNNStrategy, strategy_type=sigmoid
echo "Running Testback.py CNNStrategy,strategy_type=sigmoid"
python Testback.py CNNStrategy,strategy_type=sigmoid


python ../run_notification.py "starting Markowitz"

## - MarkowitzStrategy
echo "Running Testback.py MarkowitzStrategies"

#python Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=1 &

python Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=1 &

wait

#python Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=False,max_concentration=0.05 &
#
#python Testback.py MarkowitzStrategy,risk_constant=1,return_estimate=0.000269,vol_weighted=True,max_concentration=0.05 &
#
#wait


# - Send Email
python ../run_notification.py "Complete!"