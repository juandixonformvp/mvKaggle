library(randomForest)
library(mlbench)
library(caret)
library(e1071)
library(MASS)
library(ggparallel)
library(rpart)
library(devtools)
library(ggbiplot)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(factoextra)
library(MatchIt)
library(splitstackshape)
library(gbm)
library(xgboost)
library(glmnet)
library(MASS);
library(mda);

# setwd('home/rpchang/mvKaggle')
setwd('C:/Users/rchang/Documents/mvKaggle')
df <- read.csv('train.csv')
td <- read.csv('test.csv')

df.all <- df
# df.all <- df.all[, -which(colMeans(is.na(df.all)) > 0.000001)] #removes columns with NA values

#remove near zero-variance predictors
# remove_cols <- nearZeroVar(df.all, names = TRUE, freqCut = 2, uniqueCut = 20)
# all_cols <- names(df.all)
# df.all <- df.all[ , setdiff(all_cols, remove_cols)]

df.all$yr_renovated <- ifelse(df.all$yr_renovated > 0, 1, 0) # changing yr_renovated to binary
df.all$zipcode <- as.factor(df.all$zipcode) #zipcode is integer, but is categorical
df.all$floors <- as.factor(df.all$floors) #floors is integer, but is categorical
df.all$grade <- as.factor(df.all$grade) #grade is integer, but is categorical
df.all$yr_built <- as.factor(df.all$yr_built) #yr_built is integer, but is categorical
df.all$condition <- as.factor(df.all$condition) #grade is integer, but is categorical
df.all$waterfront <- as.factor(df.all$waterfront) #waterfront is integer, but is categorical
df.all$view <- as.factor(df.all$view) #view is integer, but is categorical
# df.all$yr_renovated <- as.factor(df.all$yr_renovated) #yr_built is integer, but is categorical
df.all$bedrooms <- as.factor(df.all$bedrooms) #bedrooms is integer, but is categorical
df.all$bathrooms <- as.factor(df.all$bathrooms) #bathrooms is integer, but is categorical

df.all <- subset(df.all, select = -c(id, date, lat, long)) #don't need "month sold" and other ID variables
x_pred <- df.all[,2:ncol(df.all)] #drop the y variable
x_pred <- x_pred[ , order(names(x_pred))]
x_train <- model.matrix( ~ .-1, x_pred)


td.all <- td[,names(x_pred)]
td.all$yr_renovated <- ifelse(td.all$yr_renovated > 0, 1, 0) # # changing yr_renovated to binary
td.all$zipcode <- as.factor(td.all$zipcode)
td.all$floors <- as.factor(td.all$floors)
td.all$grade <- as.factor(td.all$grade)
td.all$yr_built <- as.factor(td.all$yr_built)
td.all$condition <- as.factor(td.all$condition)
# td.all$yr_renovated <- as.factor(td.all$yr_renovated)
td.all$bedrooms <- as.factor(td.all$bedrooms)
td.all$bathrooms <- as.factor(td.all$bathrooms)
td.all$waterfront <- as.factor(td.all$waterfront)
td.all$view <- as.factor(td.all$view)
# options(na.action="na.pass") # the model matrix function deletes NA observations, must use "na.pass"
y_train <- model.matrix( ~ .-1, td.all)
y_train[is.na(y_train)] <- 0 # now replace NAs with 0

# problem because some categorical levels in training set are not in test set
# convert matrix to dataframe, match columns, then convert back to matrix
x_train <- as.data.frame(x_train)
x_train$bathrooms6.25 <- rep(0,nrow(x_train)) # the train dataset does not have bathrooms6.25, while test set does, so must add manually
y_train <- as.data.frame(y_train)

x_train <- x_train[,names(y_train)]

# x_train <- as.matrix(x_train)
# y_train <- as.matrix(y_train)
##########

## code to get interaction terms
f <- as.formula(y ~ .*.)
y <- df.all$price
x_interact <- model.matrix(f, x_train)[, -1]

f <- as.formula(y ~ .*.)
y <- td$id #dummy variable, just placeholder
y_interact <- model.matrix(f, y_train)[, -1]

x_interact <- as.matrix(x_interact)
y_interact <- as.matrix(y_interact)

###


fit <-glmnet(x = x_interact, y = df.all$price, alpha = 1) 
plot(fit, xvar = "lambda")

crossval <-  cv.glmnet(x = x_interact, y = df.all$price)
plot(crossval)
penalty <- crossval$lambda.min #optimal lambda
penalty #minimal shrinkage
fit1 <-glmnet(x = x_interact, y = df.all$price, alpha = 1, lambda = penalty ) #estimate the model with that
# coef(fit1)

results <- predict(object=fit1, y_interact)
pred <- cbind(td$id, as.data.frame(results))
colnames(pred)<- c("id","price")


##########MARS MODEL###############
mars.fit <- mars(x_train, df.all$price, degree = 3, prune = TRUE, forward.step = TRUE)
predictions <- predict(mars.fit, y_train)
mars_pred <- cbind(td$id, as.data.frame(predictions))
colnames(mars_pred)<- c("id","price")
##################################



# setwd('C:/Users/rchang/OneDrive - FI Consulting/Kaggle/MV_Competition')
write.csv(pred, file = "glmnet_20181112.csv", row.names=FALSE)
write.csv(mars_pred, file = "mars_20181112.csv", row.names=FALSE)
