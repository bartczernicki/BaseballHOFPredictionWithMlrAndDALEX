<a name="Title"></a>
# Demo/Hack - Baseball HOF Prediction using R Mlr & DALEX Packages

<a name="Overview"></a>
## Overview ##
This demo shows how MLB Baseball historical data from 1876 - 2017 can be used to craft a learning model to predict Hall Of Fame induction.

**mlr & DALEX Packages** allow you to craft R predictive models rapidly using sophisticated techniques such as: Bayeasian Optimization, Hyperparameter Tuning with Resampling etc.

mlr Package Information

https://mlr-org.github.io/mlr/

DALEX Package Information

https://github.com/pbiecek/DALEX

<a name="Results"></a>
## Results ##


**Predictive Model Variable Response for All-Star Appearances**
Note: As the amount of All-Star Appearances increases this increases the weight of the All-Star Appearances, thus increasing the probabiity of Hall Of Fame Induction.

![Variable Response](https://github.com/bartczernicki/BaseballHOFPredictionWithMlrAndDALEX/blob/master/Images/VariableResponse-AllStarAppearances.png)

**XgBoost Learner Prediction Plot: All-Star Apperances vs Total Bases (TB)**
Note: The XgBoost Learner (trained model) can be broken down into data visualizations that can aid in model performance analysis.  Based on this plot, note the implicit boundary of about 5,000 Total Bases & 9 All-Star Appearances as the threshold of MLB Baseball Hall Of Fame Induction.
Plot explanation:
- Dark triangle - true positive - actual HOFer predicted by the model to be in HOF
- Dark circle - true negatives - NOT a HOFer predicted by the model NOT to be in HOF
- White traingle - false positive - actual HOFer predicted by the model NOT to be in HOF
- White circle - false negative - NOT a HOFer predicted by the model to be in HOF

![Variable Response](https://github.com/bartczernicki/BaseballHOFPredictionWithMlrAndDALEX/blob/master/Images/LearnerPredictionXgBoost.png)


**Prediction Breakdown - Derek Jeter**
Note: The impact of each of the key features, which make up the final prediction of (0.89 probabilty)

![Variable Response](https://github.com/bartczernicki/BaseballHOFPredictionWithMlrAndDALEX/blob/master/Images/PredictionBreakdown-DerekJeter.png)


**Prediction Breakdown - Willie Mays**
Note: The impact of each of the key features, which make up the final prediction of (0.97 probabilty)

![Predictin Breakdown](https://github.com/bartczernicki/BaseballHOFPredictionWithMlrAndDALEX/blob/master/Images/PredictionBreakdown-WillieMays.png)
