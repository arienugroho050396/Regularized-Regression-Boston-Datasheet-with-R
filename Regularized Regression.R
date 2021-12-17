# Description: Assignment Day 20 - Regularized Regression
# Author: Mochammad Arie Nugroho
# Date: December 2021



install.packages("glmnet")

#import data
df <- read.csv("boston.csv")
View(df)


## 1. Split data into 3 parts: train - validation - test
library(caTools)
set.seed(123)
sample <- sample.split(df$medv, SplitRatio = .80)
pre_train <- subset(df, sample == TRUE)
sample_train <- sample.split(pre_train$medv, SplitRatio = .80)

# train-validation data
train <- subset(pre_train, sample_train == TRUE)
validation <- subset(pre_train, sample_train == FALSE)

# test data
test <- subset(df, sample == FALSE)



## 2. Draw correlation plot on training data and perform feature selection on highly correlated features 
# correlation study
library(psych)
pairs.panels(train, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
) # correlated features: zn, rm, black. Choose: rm

# drop correlated columns
library(dplyr)
drop_cols <- c('zn',
               'black')

train <- train %>% select(-drop_cols) #drop column data train 
validation <-  validation %>% select(-drop_cols) #drop column data validation 
test <- test %>% select(-drop_cols) #drop column data test



## 3. Fit models on training data (lambdas = [0.01, 0.1, 1, 10]) 
###### ridge regression (alpha = 0)
x <- model.matrix(medv ~ ., train)[,-1]
y <-  train$medv

library(glmnet)
ridge_reg_pointzeroone <- glmnet(x, y, alpha = 0, lambda = 0.01)
coef(ridge_reg_pointzeroone) #lamdas 0.01

ridge_reg_pointone <- glmnet(x, y, alpha = 0, lambda = 0.1)
coef(ridge_reg_pointone) #lamdas 0.1

ridge_reg_one <- glmnet(x, y, alpha = 0, lambda = 1)
coef(ridge_reg_pointone) #lamdas 1

ridge_reg_ten <- glmnet(x, y, alpha = 0, lambda = 10)
coef(ridge_reg_ten) #lamdas 10

###### LASSO (alpha = 1)
lasso_reg_pointzeroone <- glmnet(x, y, alpha = 1, lambda = 0.01)
coef(lasso_reg_pointzeroone) 

lasso_reg_pointone <- glmnet(x, y, alpha = 1, lambda = 0.1)
coef(lasso_reg_pointone) 

lasso_reg_one <- glmnet(x, y, alpha = 1, lambda = 1)
coef(lasso_reg_pointone)

lasso_reg_ten <- glmnet(x, y, alpha = 1, lambda = 10)
coef(lasso_reg_ten)



## 4. Choose the best lambda from the validation set
# Make predictions on the validation data
x_validation <- model.matrix(medv ~., validation)[,-1]
y_validation <- validation$medv
###### ridge regression (alpha = 0) 
RMSE_ridge_pointzeroone <- sqrt(mean((y_validation - predict(ridge_reg_pointzeroone, x_validation))^2))
RMSE_ridge_pointzeroone # 4.314175 -> best lambda

RMSE_ridge_pointone <- sqrt(mean((y_validation - predict(ridge_reg_pointone, x_validation))^2))
RMSE_ridge_pointone # 4.318867

RMSE_ridge_one <- sqrt(mean((y_validation - predict(ridge_reg_one, x_validation))^2))
RMSE_ridge_one # 4.407768

RMSE_ridge_ten <- sqrt(mean((y_validation - predict(ridge_reg_ten, x_validation))^2))
RMSE_ridge_ten # 5.393207

# Best model's coefficients
# recall the best model --> ridge_reg_pointzeroone
coef(ridge_reg_pointzeroone)
#interpretation:
# An increase of 1 point in Number rooms average(rm), while the other features are kept fixed, is associated with an increase of 4.369 point in Housing Price (medv).

###### LASSO regression (alpha = 0) 
RMSE_lasso_pointzeroone <- sqrt(mean((y_validation - predict(lasso_reg_pointzeroone, x_validation))^2))
RMSE_lasso_pointzeroone # 4.311475 -> best lambda

RMSE_lasso_pointone <- sqrt(mean((y_validation - predict(lasso_reg_pointone, x_validation))^2))
RMSE_lasso_pointone # 4.359183

RMSE_lasso_one <- sqrt(mean((y_validation - predict(lasso_reg_one, x_validation))^2))
RMSE_lasso_one # 4.947253

RMSE_lasso_ten <- sqrt(mean((y_validation - predict(lasso_reg_ten, x_validation))^2))
RMSE_lasso_ten # 9.371755

# Best model's coefficients
# recall the best model --> lasso_reg_pointzeroone
coef(lasso_reg_pointzeroone)
# interpretation:
# An increase of 1 point in Number rooms average(rm), while the other features are kept fixed, is associated with an increase of 4.389 point in Housing Price (medv).




## 5. Evaluate the best models on the test data (+ interpretation)
x_test <- model.matrix(medv ~., test)[,-1]
y_test <- test$medv

### ridge
# using the best model --> ridge_reg_pointzeroone

# RMSE
RMSE_ridge_best <- sqrt(mean((y_test - predict(ridge_reg_pointzeroone, x_test))^2))
RMSE_ridge_best
# interpretation: The standard deviation of prediction errors is 6.848 i.e. from the regression line, the residuals mostly deviate between +- 6.848

# MAE
MAE_ridge_best <- mean(abs(y_test-predict(ridge_reg_pointzeroone, x_test)))
MAE_ridge_best
# interpretation: On average, our prediction deviates the true medv by 4.048

# MAPE
MAPE_ridge_best <- mean(abs((predict(ridge_reg_pointzeroone, x_test) - y_test))/y_test)
MAPE_ridge_best
# interpretation: Moreover, this 4.048 is equivalent to 18.6% deviation relative to the true medv

### LASSO
# using the best model --> lasso_reg_pointzeroone

# RMSE
RMSE_lasso_best <- sqrt(mean((y_test - predict(lasso_reg_pointzeroone, x_test))^2))
RMSE_lasso_best
# interpretation:The standard deviation of prediction errors is 6.854 i.e. from the regression line, the residuals mostly deviate between +- 6.854

# MAE
MAE_lasso_best <- mean(abs(y_test-predict(lasso_reg_pointzeroone, x_test)))
MAE_lasso_best
# interpretation:On average, our prediction deviates the true medv by 4.038

# MAPE
MAPE_lasso_best <- mean(abs((predict(lasso_reg_pointzeroone, x_test) - y_test))/y_test)
MAPE_lasso_best
# interpretation: Moreover, this 4.038 is equivalent to 18.5% deviation relative to the true medv



