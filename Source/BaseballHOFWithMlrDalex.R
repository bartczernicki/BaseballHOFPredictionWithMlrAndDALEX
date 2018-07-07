# 1) Install & Load Packages
if('DALEX' %in% rownames(installed.packages()) == FALSE) {install.packages('DALEX')}
if('mlr' %in% rownames(installed.packages()) == FALSE) {install.packages('mlr')}
if('xgboost' %in% rownames(installed.packages()) == FALSE) {install.packages('xgboost')}

# Load the libraries
library(DALEX)
library(mlr)
library(xgboost)

# 2) Setup the training & validation data
wdPath = "/Users/bartczernicki-msft/Desktop/SourceCode/BaseballHOFDalex"
setwd(wdPath)

trainingData <- read.csv(file="HOFTrainingWithHeader.csv", header=TRUE, sep=",")
validationData <- read.csv(file="HOFValidationWithHeader.csv", header=TRUE, sep=",")
combinedData <- rbind(trainingData, validationData)
trainingData$InductedToHallOfFame <- factor(ifelse(trainingData$InductedToHallOfFame =='TRUE', 1, 0))
validationData$InductedToHallOfFame <- factor(ifelse(validationData$InductedToHallOfFame =='TRUE', 1, 0))
combinedData$InductedToHallOfFame <- factor(ifelse(combinedData$InductedToHallOfFame =='TRUE', 1, 0))
trainingData <- subset(trainingData, select = -c(FullPlayerName, LastYearPlayed, ID))
validationData <- subset(validationData, select = -c(FullPlayerName, LastYearPlayed, ID))
combinedData <- subset(combinedData, select = -c(FullPlayerName, LastYearPlayed, ID))
nrow(trainingData)
nrow(validationData)
nrow(combinedData)

trainingData$InductedToHallOfFame <- factor(trainingData$InductedToHallOfFame)
validationData$InductedToHallOfFame <- factor(validationData$InductedToHallOfFame)

# Check Data
head(validationData, 15)


# 3) Machine Learning

# Create (binary) classification tasks
classif_task <- makeClassifTask(id = "class1", data = trainingData, target = "InductedToHallOfFame", positive = 1)
classif_task_combined <- makeClassifTask(id = "class2", data = combinedData, target = "InductedToHallOfFame", positive = 1)

# Random Forest, GLM, SVM, XgBoost
classif_lrn_rf <- makeLearner("classif.randomForest", predict.type = "prob")
classif_lrn_glm <- makeLearner("classif.binomial", predict.type = "prob")
classif_lrn_svm <- makeLearner("classif.ksvm", predict.type = "prob")
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob")
xgboostParams <- list(nrounds = 400, max_depth = 30, nthread = 4, max_delta_step = 6,
                      num_parallel_tree = 4, eta=0.02, gamma = 1)
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob", par.vals = xgboostParams)

?configureMlr
getMlrOptions()
classif_rf <- train(classif_lrn_rf, classif_task)
classif_glm <- train(classif_lrn_glm, classif_task)
classif_svm <- train(classif_lrn_svm, classif_task)
classif_xgboost <- train(classif_lrn_xgboost, classif_task)
classif_xgboostTwo <- train(classif_lrn_xgboost, classif_task_combined)

# Pred
predRf = predict(classif_rf, newdata = validationData)
predGlm = predict(classif_glm, newdata = validationData)
predSvm = predict(classif_svm, newdata = validationData)
predXgBoost = predict(classif_xgboost, newdata = validationData)
predXgBoostTwo = predict(classif_xgboostTwo, newdata = validationData)

# Test Perf
listMeasures("classif")
getDefaultMeasure(classif_lrn_rf)
performanceMetrics <- list(tnr, tpr, lsr, f1, mmce, tp, acc, fdr, kappa, ppv)
performance(predRf, measures = performanceMetrics)
performance(predGlm, measures = performanceMetrics)
performance(predSvm, measures = performanceMetrics)
performance(predXgBoost, measures = performanceMetrics)

d <- generateThreshVsPerfData(predRf, measures = list(tpr, ppv, tp))
plotThreshVsPerf(d)
threshholdvsPerf_XgBoost <- generateThreshVsPerfData(predXgBoost, measures = list(tpr, ppv, tp, mmce))
plotThreshVsPerf(threshholdvsPerf_XgBoost)
rocMeasRf <- calculateROCMeasures(predRf)
rocMeasRf
rocMeasXgBoost <- calculateROCMeasures(predXgBoost)
rocMeasXgBoost

# Plot Learner Prediction - All-Star Appearances vs TB
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task_training, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_svm, task = classif_task_training, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task_training, features = c("AllStarAppearances", "TB"), cv = 10)

plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("H", "HR"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "HR"), cv = 10)

plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "SluggingPct"), cv = 10)

plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "GoldGloves"), cv = 10)

plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("MVPs", "H"), cv = 10)

# Test Remove
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task_combined, features = c("AllStarAppearances", "TB"), cv = 10)

# Plot ROC Curves
df = generateThreshVsPerfData(list(xgBoost = predXgBoost, rf = predRf, glm = predGlm, xgboost = predXgBoost), 
                              measures = list(fpr, tpr)) 
plotROCCurves(df)



# Hyper Parameters
getParamSet("classif.xgboost")
getParamSet("classif.randomForest")
discrete_ps <- makeParamSet(
  makeDiscreteParam("ntree", values = c(1, 50, 100, 150, 400, 600)),
  makeLogicalParam("replace")
)
tuneControl <- makeTuneControlRandom(maxit = 10L)
rdesc = makeResampleDesc("CV", iters = 3L)
resampling <- rdesc

resultsTuning <- tuneParams("classif.randomForest", task = classif_task, resampling = rdesc,
                            par.set = discrete_ps, 
                            measures = list(ppv, setAggregation(ppv, test.sd)),
                            control = tuneControl)

# Custom Holdout
rin = makeFixedHoldoutInstance(train.inds = 1:6258, test.inds = 6259:9385, size=9385)
rin
resultsTuning <- tuneParams(#"classif.randomForest"
                            classif_lrn_rf, task = classif_task_combined, resampling = rdesc,
                            par.set = discrete_ps, 
                            measures = list(auc, setAggregation(auc, test.sd)),
                            control = tuneControl)


y_test <- as.numeric(as.character(validationData$InductedToHallOfFame))
custom_predict_classif <- function(object, newdata) {pred <- predict(object, newdata=newdata)
{
  response <- pred$data[,3]
  return(response)}  
}


explainer_classif_rf <- DALEX::explain(classif_rf, data=validationData, y=y_test, label= "rf", predict_function = custom_predict_classif)
explainer_classif_glm <- DALEX::explain(classif_glm, data=validationData, y=y_test, label="glm", predict_function = custom_predict_classif)
explainer_classif_svm <- DALEX::explain(classif_svm, data=validationData, y=y_test, label ="svm", predict_function = custom_predict_classif)
explainer_classif_xgboost <- DALEX::explain(classif_xgboost, data=validationData, y=y_test, label ="xgboost", predict_function = custom_predict_classif)

# Model Performance
mp_classif_rf <- model_performance(explainer_classif_rf)
mp_classif_glm <- model_performance(explainer_classif_glm)
mp_classif_svm <- model_performance(explainer_classif_svm)
mp_classif_xgboost <- model_performance(explainer_classif_xgboost)
plot(mp_classif_rf, mp_classif_glm, mp_classif_svm, mp_classif_xgboost)

# Variable Importance
vi_classif_rf <- variable_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
vi_classif_glm <- variable_importance(explainer_classif_glm, loss_function = loss_root_mean_square)
vi_classif_svm <- variable_importance(explainer_classif_svm, loss_function = loss_root_mean_square)
vi_classif_xgboost <- variable_importance(explainer_classif_xgboost, loss_function = loss_root_mean_square)
plot(vi_classif_rf, vi_classif_glm, vi_classif_svm, vi_classif_xgboost)
plot(vi_classif_xgboost)

# Variable Response - All Star Appearances Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "AllStarAppearances", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "AllStarAppearances", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "AllStarAppearances", type = "pdp")
pdp_classif_xgboost  <- variable_response(explainer_classif_xgboost, variable = "AllStarAppearances", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm, pdp_classif_xgboost)

# Variable Response - Random Forest Model - HR Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "HR", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)
# Variable Response - Random Forest Model - MVP Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "MVPs", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "MVPs", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "MVPs", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)
# Variable Response - Random Forest Model - TB (Total Bases) Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "TB", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "TB", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "TB", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)
# Variable Response - Random Forest Model - GoldGloves Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "GoldGloves", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "GoldGloves", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "GoldGloves", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)


# Prediction Breakdown - Ichiro Suzuki
newPred31 <- validationData[31,]
newPredDownRf31 <- prediction_breakdown(explainer_classif_rf, observation = newPred31)
newPredDownRf31
plot(newPredDownRf31)

# Prediction Breakdown - Kirby Puckett
newPred31 <- validationData[22,]
newPredDownRf31 <- prediction_breakdown(explainer_classif_rf, observation = newPred31)
newPredDownRf31
plot(newPredDownRf31)


# Prediction Breakdown - Willie Mays
newPred12 <- validationData[12,]
newPredDownRf12 <- prediction_breakdown(explainer_classif_rf, observation = newPred12)
newPredDownRf12
plot(newPredDownRf12)

# Prediction Breakdown - Derek Jeter
newPred29 <- validationData[29,]
newPredDownRf29 <- prediction_breakdown(explainer_classif_rf, observation = newPred29)
newPredDownRf29
plot(newPredDownRf29)

# Prediction Breakdown - Tony Perez
newPred18 <- validationData[18,]
newPredDownRf18 <- prediction_breakdown(explainer_classif_rf, observation = newPred18)
newPredDownRf18
plot(newPredDownRf18)

