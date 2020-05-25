# TESTED ON CRAN R 3.6.1
# Last Updated: 05/25/2020

# Check your R environment version
# Should be close to the tested version(s)
version



# 1) INSTALL & LOAD PACKAGES

# a) INSTALL Core Modeling Packages
install.packages("corrplot")
install.packages('Hmisc')
install.packages('mlr')
install.packages('kernlab')
install.packages('randomForest')
install.packages('xgboost')
# b) ML Orchestration/Explainer/Helper Packages
install.packages('ceterisParibus')
install.packages('breakDown')
install.packages('DALEX')
install.packages('ingredients')
install.packages('jsonlite')
install.packages('pdp')

# b) LOAD Packages
#Core Statistical Packages
library(corrplot)
library(ggplot2)
library(reshape2)
library(Hmisc)
library(stats)
# Core Modeling Packages
library(mlr)                # Used for training/orchestrating models
library(kernlab)
library(randomForest)       # Used For RandomForest classifier
library(xgboost)            # Used for XgBoost classifier
# ML Orchestration/Explainer/Helper Packages
library(ceterisParibus)     # Used for What-If Charts
library(breakDown)          # Used for Model Explainers
library(ingredients)        # Used for Model Explainers
library(DALEX)              # Used for Model Explainers
library(jsonlite)           # Used for Model Explainers
library(pdp)                # Used for Model Explainers

# c) Check versions (optional)
# Tested on..
# mlr_2.15.0
# ceterisParibus_0.3.1
# DALEX_0.4.9
# breakDown_0.1.6
sessionInfo()

# 2) SETUP TRAINING, VALIDATION & COMBINED DATA SETS

# a) Set working directory
getwd()
# Working Directory format for macOS
wdPath <- "/Users/bartczernicki-msft/Desktop/SourceCode/BaseballHOFDalex"
# Working Directory format for Windows OS
# wdPath <- "C:\\Users\\bart\\Downloads\\BaseballHOFPredictionWithMlrAndDALEX-master\\BaseballHOFPredictionWithMlrAndDALEX-master\\Source"
setwd(wdPath)


# b) Load Training Data CSV Files
# Custom sampling was used to build this, using simple techniques
# Note: Full proper model leverages re-sampling based on baseball eras, adjustments to the rules as well as player position
trainingData <- read.csv(file="BaseballHOFTrainingv2.csv", header=TRUE, sep=",")
validationData <- read.csv(file="BaseballHOFTestv2.csv", header=TRUE, sep=",")
combinedData <- rbind(trainingData, validationData)

# c) Change InductedToHallOfFame as a 1/0 binary factor
trainingData$InductedToHallOfFame <- factor(ifelse(trainingData$InductedToHallOfFame =='TRUE', 1, 0))
validationData$InductedToHallOfFame <- factor(ifelse(validationData$InductedToHallOfFame =='TRUE', 1, 0))
combinedData$InductedToHallOfFame <- factor(ifelse(combinedData$InductedToHallOfFame =='TRUE', 1, 0))

# d) Copy data set to make predictions in bottom of script, before removing player Name and ID
# This is so we can easily lookup players by ID or FullPlayerName later
fullPlayerData <- combinedData

# e) Remove ID columns (FullPLayerName, LastYearPlayed, ID) not needed for training classifier models
trainingData <- subset(trainingData, select = -c(FullPlayerName, LastYearPlayed, ID, OnHallOfFameBallot))
validationData <- subset(validationData, select = -c(FullPlayerName, LastYearPlayed, ID, OnHallOfFameBallot))
combinedData <- subset(combinedData, select = -c(FullPlayerName, LastYearPlayed, ID, OnHallOfFameBallot))
nrow(trainingData)
nrow(validationData)
nrow(combinedData)

# f) Check Validation Data
head(validationData, 15)

# Corelation Matrix
res2 <-cor.test(combinedData$MVPs, combinedData$TotalPlayerAwards,  method = "spearman")
cormatrix = rcorr(as.matrix(combinedData), type='spearman')
cormatrix
cordata = melt(cormatrix$r)
ggplot(cordata, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + xlab("") + ylab("")

# 3) MACHINE LEARNING - TRAINING

# a) Set the seed explicitly for training reproducability
set.seed(12345)

# b) Create (binary) classification tasks (MLR package construct)
classif_task <- makeClassifTask(id = "class1", data = trainingData, target = "InductedToHallOfFame", positive = 1)
classif_task_combined <- makeClassifTask(id = "class2", data = combinedData, target = "InductedToHallOfFame", positive = 1)

# C) Make Learners - Random Forest, GLM, XgBoost (MLR package construct)
# Get a list of available learners (will list available and what learners are installed)
# Note: Any of the learners below can be used
listLearners("classif", properties = c("twoclass", "prob"))

# Make different learners (simple - using default hyperparameters)
classif_lrn_rf <- makeLearner("classif.randomForest", predict.type = "prob")
classif_lrn_glm <- makeLearner("classif.binomial", predict.type = "prob")
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob")
# Note: Ignore warning about NAs (harmless warning)

# d) Traing the classification models using learners (and their accompanying parameters)
# Run this command for MLR configuration: ?configureMlr
# Get MLR options getMlrOptions()
classif_rf <- train(classif_lrn_rf, classif_task)
classif_glm <- train(classif_lrn_glm, classif_task)
classif_xgboost <- train(classif_lrn_xgboost, classif_task)

# e) Make predictions on validation data
predRf = predict(classif_rf, newdata = validationData)
predGlm = predict(classif_glm, newdata = validationData)
predXgBoost = predict(classif_xgboost, newdata = validationData)



# 4) MACHINE LEARNING - EVALUATION

# a) Test Performance of classifiers (models) using various performance metrics

# You can get a list of available measures
listMeasures("classif")
getDefaultMeasure(classif_lrn_rf)
# View performance metrics
performanceMetrics <- list(tnr, tpr, lsr, f1, mmce, tp, acc, fdr, kappa, ppv)
performance(predRf, measures = performanceMetrics)
performance(predGlm, measures = performanceMetrics)
performance(predXgBoost, measures = performanceMetrics)

# b) Build a list of important measures to analyze visualy, generate performance data, plot performance data
relevantPerformanceMetrics <- list(tpr, ppv, tp, mmce)

threshholdvsPerf_Rf <- generateThreshVsPerfData(predRf, measures = relevantPerformanceMetrics)
threshholdvsPerf_glm <- generateThreshVsPerfData(predGlm, measures = relevantPerformanceMetrics)
threshholdvsPerf_XgBoost <- generateThreshVsPerfData(predXgBoost, measures = relevantPerformanceMetrics)
# Plot performance metrics
plotThreshVsPerf(threshholdvsPerf_Rf)
plotThreshVsPerf(threshholdvsPerf_glm)
plotThreshVsPerf(threshholdvsPerf_XgBoost)

# c) Calculcate & Show ROC Measures / Confusion Matrix
rocMeasures_Rf <- calculateROCMeasures(predRf)
rocMeasures_Rf
rocMeasures_glm <- calculateROCMeasures(predGlm)
rocMeasures_glm
rocMeasures_XgBoost <- calculateROCMeasures(predXgBoost)
rocMeasures_XgBoost

# Plot ROC Curves for the trained classifiers/models
rocCurvesData <- generateThreshVsPerfData(list(rf = predRf, glm = predGlm, xgboost = predXgBoost), 
                                          measures = list(fpr, tpr)) 
plotROCCurves(rocCurvesData)

# d) Plot Learner Prediction - All-Star Appearances vs 

# AllStarAppearances vs TB (Total Bases)
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)

# H vs HR
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("H", "HR"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("H", "HR"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "HR"), cv = 10)

# H vs GoldGloves
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("H", "GoldGloves"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("H", "GoldGloves"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "GoldGloves"), cv = 10)

# MVP vs H
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("MVPs", "H"), cv = 10)



#4) MACHINE LEARNING - MODEL EXPLAINERS
# Uses the DALEX r package

# a) Build a helper function to extract prediction data, required for DALEX explainers
y_test <- as.numeric(as.character(validationData$InductedToHallOfFame))
custom_predict_classif <- function(object, newdata)
  {
    pred <- predict(object, newdata=newdata)
      {
        response <- pred$data[,3]
        return(response)
      }  
  }

# b) Build model exmplainers for each of the built models
explainer_classif_rf <- DALEX::explain(classif_rf, data=validationData, y=y_test, label= "rf", predict_function = custom_predict_classif)
explainer_classif_glm <- DALEX::explain(classif_glm, data=validationData, y=y_test, label="glm", predict_function = custom_predict_classif)
explainer_classif_xgboost <- DALEX::explain(classif_xgboost, data=validationData, y=y_test, label ="xgboost", predict_function = custom_predict_classif)

# c) Plot Model Performance
?model_performance
mp_classif_rf <- model_performance(explainer_classif_rf)
mp_classif_glm <- model_performance(explainer_classif_glm)
mp_classif_xgboost <- model_performance(explainer_classif_xgboost)
plot(mp_classif_rf, mp_classif_glm, mp_classif_xgboost)

# d) Plot Feature Importance - Individually & Collectively
vi_classif_rf <- ingredients::feature_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
vi_classif_glm <- ingredients::feature_importance(explainer_classif_glm, loss_function = loss_root_mean_square)
vi_classif_xgboost <- ingredients::feature_importance(explainer_classif_xgboost, loss_function = loss_root_mean_square)
plot(vi_classif_rf, vi_classif_glm, vi_classif_xgboost) #Plot all three feature importance
plot(vi_classif_glm) #Plot GLM feature importance
plot(vi_classif_rf) #Plot RF feature importance
plot(vi_classif_xgboost) #Plot XgBoost feature importance

# Variable Response - All Star Appearances Feature
vr_AllStarAppearances_rf = partial_dependency(explainer_classif_rf, variables = "AllStarAppearances")
vr_AllStarAppearances_glm  <- partial_dependency(explainer_classif_glm, variables = "AllStarAppearances")
vr_AllStarAppearances_xgboost  <- partial_dependency(explainer_classif_xgboost, variables = "AllStarAppearances")
plot(vr_AllStarAppearances_rf, vr_AllStarAppearances_glm, vr_AllStarAppearances_xgboost)

# Variable Response - Random Forest Model - HR Feature
vr_hr_rf  <- partial_dependency(explainer_classif_rf, variables = "HR")
vr_hr_glm  <- partial_dependency(explainer_classif_glm, variables = "HR")
vr_hr_xgboost  <- partial_dependency(explainer_classif_xgboost, variables = "HR")
plot(vr_hr_rf, vr_hr_glm, vr_hr_xgboost)

# Variable Response - Random Forest Model - MVP Feature
vr_mvp_rf  <- partial_dependency(explainer_classif_rf, variables = "MVPs")
vr_mvp_glm  <- partial_dependency(explainer_classif_glm, variables = "MVPs")
vr_mvp_xgboost  <- partial_dependency(explainer_classif_xgboost, variables = "MVPs")
plot(vr_mvp_rf, vr_mvp_glm, vr_mvp_xgboost)

# Variable Response - Random Forest Model - TB (Total Bases) Feature
vr_tb_rf  <- partial_dependency(explainer_classif_rf, variables = "TB")
vr_tb_glm  <- partial_dependency(explainer_classif_glm, variables= "TB")
vr_tb_xgboost  <- partial_dependency(explainer_classif_xgboost, variables = "TB")
plot(vr_tb_rf, vr_tb_glm, vr_tb_xgboost)

# Variable Response - Random Forest Model - GoldGloves Feature
vr_goldGloves_rf  <- variable_response(explainer_classif_rf, variable = "GoldGloves", type = "pdp")
vr_goldGloves_glm  <- variable_response(explainer_classif_glm, variable = "GoldGloves", type = "pdp")
vr_goldGloves_xgboost  <- variable_response(explainer_classif_xgboost, variable = "GoldGloves", type = "pdp")
plot(vr_goldGloves_rf, vr_goldGloves_glm, vr_goldGloves_xgboost)

#5) MACHINE LEARNING - INDIVIDUAL PREDICTIONS

# Prediction Breakdown - Ichiro Suzuki
ichiroSuzukiData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Ichiro Suzuki",], 1)
prediction_breakdown_IchiroSuzuki <- prediction_breakdown(explainer_classif_rf, observation = ichiroSuzukiData)
prediction_breakdown_IchiroSuzuki
plot(prediction_breakdown_IchiroSuzuki)
# What-If - Ichiro Suzuki
whatIf_rf_ichiroSuzuki <- what_if(explainer_classif_rf, observation = ichiroSuzukiData,
                                  selected_variables = c("H", "TB", "AllStarAppearances"))
whatIf_rf_ichiroSuzuki
plot(whatIf_rf_ichiroSuzuki)

# Prediction Breakdown - Kirby Puckett
kirbyPuckettData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Kirby Puckett",], 1)
prediction_breakdown_KirbyPuckett = prediction_breakdown(explainer_classif_rf, observation = kirbyPuckettData)
plot(prediction_breakdown_KirbyPuckett)
# What-If - Kirby Puckett
whatIf_rf_kirbyPuckett <- what_if(explainer_classif_rf, observation = kirbyPuckettData,
                                  selected_variables = c("H", "TB", "AllStarAppearances"))
whatIf_rf_kirbyPuckett
plot(whatIf_rf_kirbyPuckett)

# Prediction Breakdown - Willie Mays
willieMaysData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Willie Mays",], 1)
prediction_breakdown_WillieMays = prediction_breakdown(explainer_classif_rf, observation = willieMaysData)
plot(prediction_breakdown_WillieMays)

# Prediction Breakdown - Derek Jeter
derrekJeterData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Derek Jeter",], 1)
prediction_breakdown_DerekJeter = prediction_breakdown(explainer_classif_rf, observation = derrekJeterData)
plot(prediction_breakdown_DerekJeter)

# Prediction Breakdown - Chase Utley
chaseUtleyData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Chase Utley",], 1)
prediction_breakdown_ChaseUtley = prediction_breakdown(explainer_classif_rf, observation = chaseUtleyData)
plot(prediction_breakdown_ChaseUtley)

# Prediction Breakdown - Mike Trout
mikeTroutData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Mike Trout",], 1)
prediction_breakdown_mikeTroutData = prediction_breakdown(explainer_classif_rf, observation = mikeTroutData)
plot(prediction_breakdown_mikeTroutData)
# What-If - Mike Trout
whatIf_rf_mikeTrout <- what_if(explainer_classif_rf, observation = mikeTroutData,
                               selected_variables = c("H", "TB", "AllStarAppearances"))
plot(whatIf_rf_mikeTrout)

# Prediction Breakdown - Jimmy Rollins
jimmyRollinsData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jimmy Rollins",], 1)
prediction_breakdown_JimmyRollins = prediction_breakdown(explainer_classif_rf, observation = jimmyRollinsData)
plot(prediction_breakdown_JimmyRollins)

# Prediction Breakdown - Jeff Kent
jeffKentData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jeff Kent",], 1)
prediction_breakdown_JeffKent = prediction_breakdown(explainer_classif_rf, observation = jeffKentData)
plot(prediction_breakdown_JeffKent)

#Modified Prediction
jeffKentData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jeff Kent",], 1)
prediction_breakdown_JeffKent = prediction_breakdown(explainer_classif_rf, observation = jeffKentData)
plot(prediction_breakdown_JeffKent)
attributes(prediction_breakdown_JeffKent)


# Selected 2nd basemen

# Prediction Breakdown - Rod Carew
rodCarewData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Rod Carew",], 1)
prediction_breakdown_RodCarew = prediction_breakdown(explainer_classif_rf, observation = rodCarewData)
plot(prediction_breakdown_RodCarew)

# Prediction Breakdown - Roberto Alomar
robertoAlomarData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Roberto Alomar",], 1)
prediction_breakdown_RobertoAlomar = prediction_breakdown(explainer_classif_rf, observation = robertoAlomarData)
plot(prediction_breakdown_RobertoAlomar)

# Prediction Breakdown - Roberto Alomar
ryneSandbergData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Ryne Sandberg",], 1)
prediction_breakdown_RyneSandberg = prediction_breakdown(explainer_classif_rf, observation = ryneSandbergData)
plot(prediction_breakdown_RyneSandberg)

# Prediction Breakdown - Roberto Alomar
robinsonCanoData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Robinson Cano",], 1)
prediction_breakdown_RobinsonCano = prediction_breakdown(explainer_classif_rf, observation = robinsonCanoData)
plot(prediction_breakdown_RobinsonCano)
