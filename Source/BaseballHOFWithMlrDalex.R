# TESTED ON MRO 3.5.0 & CRAN R 3.5.1
# Check your R environment version
version



# 1) INSTALL & LOAD PACKAGES
# a) INSTALL Core Modeling Packages
if('ceterisParibus' %in% rownames(installed.packages()) == FALSE) {install.packages('ceterisParibus')}
if('kernlab' %in% rownames(installed.packages()) == FALSE) {install.packages('kernlab')}
if('randomForest' %in% rownames(installed.packages()) == FALSE) {install.packages('randomForest')}
if('xgboost' %in% rownames(installed.packages()) == FALSE) {install.packages('xgboost')}
# ML Orchestration/Helper Packages
if('DALEX' %in% rownames(installed.packages()) == FALSE) {install.packages('DALEX')}
if('mlr' %in% rownames(installed.packages()) == FALSE) {install.packages('mlr')}

# b) LOAD Package
# Core Modeling Packages
library(ceterisParibus)     # Used for What-If Charts
library(kernlab)
library(randomForest)       # Used For RandomForest classifier
library(xgboost)            # Used for XgBoost classifier
# Core Modeling Packages
library(DALEX)              # Used for Model Explainers
library(mlr)                # Used for training/orchestrating models

# c) Check versions (optional)
sessionInfo()

# 2) SETUP TRAINING, VALIDATION & COMBINED DATA SETS

# a) Set working directory
getwd()
wdPath <- "/Users/bartczernicki-msft/Desktop/SourceCode/BaseballHOFDalex"
# wdPath <- "C:\\Users\\bart\\Downloads\\BaseballHOFPredictionWithMlrAndDALEX-master\\BaseballHOFPredictionWithMlrAndDALEX-master\\Source"
setwd(wdPath)

# b) Load Training Data CSV Files
trainingData <- read.csv(file="HOFTrainingWithHeader.csv", header=TRUE, sep=",")
validationData <- read.csv(file="HOFValidationWithHeader.csv", header=TRUE, sep=",")
combinedData <- rbind(trainingData, validationData)

# c) Change InductedToHallOfFame as a 1/0 binary factor
trainingData$InductedToHallOfFame <- factor(ifelse(trainingData$InductedToHallOfFame =='TRUE', 1, 0))
validationData$InductedToHallOfFame <- factor(ifelse(validationData$InductedToHallOfFame =='TRUE', 1, 0))
combinedData$InductedToHallOfFame <- factor(ifelse(combinedData$InductedToHallOfFame =='TRUE', 1, 0))

# d) Copy data set to make predictions in bottom of script, before removing player Name and ID
# This is so we can easily lookup players by ID or FullPlayerName later
fullPlayerData <- combinedData

# e) Remove ID columns (FullPLayerName, LastYearPlayed, ID) not needed for training classifier models
trainingData <- subset(trainingData, select = -c(FullPlayerName, LastYearPlayed, ID))
validationData <- subset(validationData, select = -c(FullPlayerName, LastYearPlayed, ID))
combinedData <- subset(combinedData, select = -c(FullPlayerName, LastYearPlayed, ID))
nrow(trainingData)
nrow(validationData)
nrow(combinedData)

# f) Check Validation Data
head(validationData, 15)



# 3) MACHINE LEARNING - TRAINING

# a) Set the seed explicitly for training reproducability
set.seed(12345)

# b) Create (binary) classification tasks (MLR package construct)
classif_task <- makeClassifTask(id = "class1", data = trainingData, target = "InductedToHallOfFame", positive = 1)
classif_task_combined <- makeClassifTask(id = "class2", data = combinedData, target = "InductedToHallOfFame", positive = 1)

# C) Make Learners - Random Forest, GLM, XgBoost (MLR package construct)
# Get a list of available learners (will list available and what learners are installed)
listLearners("classif", properties = c("twoclass", "prob"))

# Make different learners (simple - using default hyperparameters)
classif_lrn_rf <- makeLearner("classif.randomForest", predict.type = "prob")
classif_lrn_glm <- makeLearner("classif.binomial", predict.type = "prob")
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob")

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
predXgBoostTwo = predict(classif_xgboostTwo, newdata = validationData)



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
custom_predict_classif <- function(object, newdata) {pred <- predict(object, newdata=newdata)
{
  response <- pred$data[,3]
  return(response)}  
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

# d) Plot Variable Importance - Individually & Collectively
vi_classif_rf <- variable_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
vi_classif_glm <- variable_importance(explainer_classif_glm, loss_function = loss_root_mean_square)
vi_classif_xgboost <- variable_importance(explainer_classif_xgboost, loss_function = loss_root_mean_square)
plot(vi_classif_rf, vi_classif_glm, vi_classif_xgboost)
plot(vi_classif_xgboost)

# Variable Response - All Star Appearances Feature
vr_AllStarAppearances_rf  <- variable_response(explainer_classif_rf, variable = "AllStarAppearances", type = "pdp")
vr_AllStarAppearances_glm  <- variable_response(explainer_classif_glm, variable = "AllStarAppearances", type = "pdp")
vr_AllStarAppearances_xgboost  <- variable_response(explainer_classif_xgboost, variable = "AllStarAppearances", type = "pdp")
plot(vr_AllStarAppearances_rf, vr_AllStarAppearances_glm, vr_AllStarAppearances_xgboost)

# Variable Response - Random Forest Model - HR Feature
vr_hr_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "pdp")
vr_hr_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "pdp")
vr_hr_xgboost  <- variable_response(explainer_classif_xgboost, variable = "HR", type = "pdp")
plot(vr_hr_rf, vr_hr_glm, vr_hr_xgboost)

# Variable Response - Random Forest Model - MVP Feature
vr_mvp_rf  <- variable_response(explainer_classif_rf, variable = "MVPs", type = "pdp")
vr_mvp_glm  <- variable_response(explainer_classif_glm, variable = "MVPs", type = "pdp")
vr_mvp_xgboost  <- variable_response(explainer_classif_xgboost, variable = "MVPs", type = "pdp")
plot(vr_mvp_rf, vr_mvp_glm, vr_mvp_xgboost)

# Variable Response - Random Forest Model - TB (Total Bases) Feature
vr_tb_rf  <- variable_response(explainer_classif_rf, variable = "TB", type = "pdp")
vr_tb_glm  <- variable_response(explainer_classif_glm, variable = "TB", type = "pdp")
vr_tb_xgboost  <- variable_response(explainer_classif_xgboost, variable = "TB", type = "pdp")
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
whatIf_rf_ichiroSuzuki <- ceteris_paribus(explainer_classif_rf, observation = ichiroSuzukiData, selected_variables = c("MVPs"))
whatIf_rf_ichiroSuzuki
plot(whatIf_rf_ichiroSuzuki, split = "variables", color = "variables")

# Prediction Breakdown - Kirby Puckett
kirbyPuckettData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Kirby Puckett",], 1)
prediction_breakdown_KirbyPuckett = prediction_breakdown(explainer_classif_rf, observation = kirbyPuckettData)
plot(prediction_breakdown_KirbyPuckett)
# What-If - Kirby Puckett
whatIf_rf_kirbyPuckett <- ceteris_paribus(explainer_classif_rf, observation = kirbyPuckettData)
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
whatIf_rf_mikeTrout <- ceteris_paribus(explainer_classif_rf, observation = mikeTroutData, grid_points = 1000)
whatIf_rf_mikeTrout
plot(whatIf_rf_mikeTrout)

# Prediction Breakdown - Jimmy Rollins
jimmyRollinsData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jimmy Rollins",], 1)
prediction_breakdown_JimmyRollins = prediction_breakdown(explainer_classif_rf, observation = jimmyRollinsData)
plot(prediction_breakdown_JimmyRollins)
# What-If - Jimmy Rollins
whatIf_rf_jimmyRollins <- ceteris_paribus(explainer_classif_rf, observation = jimmyRollinsData)
whatIf_glm_jimmyRollins <- ceteris_paribus(explainer_classif_glm, observation = jimmyRollinsData)
jimmyRollinsDataXgBoost <- subset(jimmyRollinsData, select = -c(FullPlayerName, LastYearPlayed, ID))
whatIf_xgBoost_jimmyRollins <- ceteris_paribus(explainer_classif_xgboost, observation = jimmyRollinsDataXgBoost)
plot(whatIf_xgBoost_jimmyRollins)

# Prediction Breakdown - Jeff Kent
jeffKentData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jeff Kent",], 1)
prediction_breakdown_JeffKent = prediction_breakdown(explainer_classif_rf, observation = jeffKentData)
plot(prediction_breakdown_JeffKent)
# What-If - Jeff Kent
whatIf_rf_jeffKent <- ceteris_paribus(explainer_classif_rf, observation = jeffKentData)
plot(whatIf_rf_jeffKent)
selectedVariables <- c('MVPs')
whatIf_rf_jeffKentAllStarApperances <- ceteris_paribus(explainer_classif_rf,
  observation = jeffKentData, selected_variables = selectedVariables, grid_points = 20)
plot(whatIf_rf_jeffKentAllStarApperances)
str(whatIf_rf_jeffKentAllStarApperances)
summary(whatIf_rf_jeffKentAllStarApperances)
whatIf_rf_jeffKentAllStarApperances

whatIf_rf_jeffKentAllStarApperances$new_x

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
