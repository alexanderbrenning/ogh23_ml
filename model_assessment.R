#####################################################
# OpenGeoHub Summer School 2023, Poznan
# Alexander Brenning
# University of Jena, Germany
####################################################
# Model assessment with spatial cross-validation
# Case study: landslides in Ecuador
####################################################

# Fewer decimal places - works better for instructor:
options(digits=4)

library("sperrorest")     # spatial CV
library("mgcv")           # GAM
library("rpart")          # CART
library("randomForest")   # Random forest

# Load the saved training/test data:
d <- readRDS("landslides.rds")

# Apply some transformations to the data:
my.trafo <- function(x) {
  # Same for >300m from deforestation:
  x$distdeforest[ x$distdeforest > 300 ] <- 300
  # ...and >300m from road:
  x$distroad[ x$distroad > 300 ] <- 300
  # Convert radians to degrees:
  x$slope <- x$slope * 180 / pi
  x$cslope <- x$cslope * 180 / pi
  # Log-transform size of catchment area - it is extremely skewed:
  x$log.carea <- log10(x$carea)
  # Plan and profile curvature have outliers -> trim both variables:
  x$plancurv[ x$plancurv < -0.2 ] <- -0.2
  x$plancurv[ x$plancurv >  0.2 ] <-  0.2
  x$profcurv[ x$profcurv < -0.04 ] <- -0.04
  x$profcurv[ x$profcurv >  0.04 ] <-  0.04
  return(x)
}

d <- my.trafo(d)

# Make sure we use the exact same formula in all models:
fo <- slides89 ~ slope + plancurv + profcurv + log.carea + cslope + distroad + distdeforest


#####################################################
# Start by exploring spatial resampling in sperrorest
#####################################################

# Resampling for non-spatial cross-validation:
# 5 folds, 2 repetitions for illustration
resamp <- partition_cv(d, nfold=5, repetition=1:2)
plot(resamp, d)

# Take a look inside:
# first repetition, second fold: row numbers 
# of observations in the test set:
idx <- resamp[["1"]][[2]]$test
# test sample that would be used in this particular 
# repetition and fold:
d[ idx , ]




# Resampling for spatial cross-validation 
# using k-means clustering of coordinates:
resamp <- partition_kmeans(d, coords = c("x","y"),
                           nfold=5, repetition=1:2)
plot(resamp, d)
# Repeat this to get different partitions (depends on
# data set; works better with nfold=10).
# Use the seed1 argument to make your partitioning
# reproducible.


p_load("rpart")
# A wrapper function that extracts the predicted probabilities from rpart predictions:
mypred_rpart <- function(object, newdata) {
  predict(object, newdata, type="prob")[,2]
}
# Control parameters for rpart:
ctrl <- rpart.control(cp=0.003)

# Perform 5-repeated 5-fold non-spatial cross-validation:
res <- sperrorest(formula = fo, data = d, 
                  coords = c("x","y"),
                 model_fun = rpart, 
                 model_args = list(control=ctrl),
                 pred_fun = mypred_rpart,
                 smp_fun = partition_cv, 
                 smp_args = list(repetition=1:5, nfold=5))
# In "real life" we should use nfold=10 and repetition=1:100 !!!
# plot(res$represampling, d) # may be too large to plot properly
summary(res$error_rep)
# More detailed, for each repetition:
summary(res$error_rep, level=1)
# Let's focus on AUROC on training and test set:
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )

# Let's get the values auroc recorded for each repetition
auroc.test <- unlist(summary(res$error_rep,level=1)[,"test_auroc"])
auroc.train <- unlist(summary(res$error_rep,level=1)[,"train_auroc"])
mean(auroc.test)
mean(auroc.train)

# We can also summarize our results as a box plot
df_auroc <- data.frame(training = auroc.train, testing = auroc.test)
boxplot(df_auroc, ylab = "AUROC", col = "lightblue", ylim = c(0.5,1))

# Perform 5-repeated 5-fold SPATIAL cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                 model_fun = rpart, 
                 model_args = list(control=ctrl), 
                 pred_fun = mypred_rpart, 
                 smp_fun = partition_kmeans, 
                 smp_args = list(repetition=1:5, nfold=5))
# Let's focus on AUROC on training and test set:
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )
# Spatial cross-validation reveals overfitting that was not detected
# by non-spatial cross-validation...


# Now let's tidy up the code and do this for the GAM, CART and random forest.
# (Pick one method in class since this is computationally intensive.)



####################################
# Generalized Additive Model
####################################

library("mgcv")

# my_gam is a wrapper function that we need because mgcv::gam() requires
# s() terms in the formula object in order to produce nonlinear
# spline smoothers...
my_gam <- function(formula, data, family = binomial, k = 4) {
  response.name <- as.character(formula)[2]
  predictor.names <- labels(terms(formula))
  categorical <- sapply(data[,predictor.names], is.logical) |
                 sapply(data[,predictor.names], is.factor)
  formula <- paste(response.name, "~",
                   paste(predictor.names[categorical], collapse = "+"),
                   paste("s(", predictor.names[!categorical], ", k=", k, ")", collapse = "+"))
  formula <- as.formula(formula)
  # Return fitted GAM model:
  gam(formula, data, family = family, select = TRUE)
}


# Check that our wrapper function is working:
# fit <- my.gam(fo,d)
# summary(fit)
# plot(fit)

# Perform 5-repeated 5-fold non-spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                  model_fun = my_gam, 
                  pred_args = list(type="response"), 
                  smp_fun = partition_cv, 
                  smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )


# Perform 5-repeated 5-fold spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                 model_fun = my_gam, 
                 pred_args = list(type="response"), 
                 smp_fun = partition_kmeans, smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )



####################################
# Classification tree
####################################

library("rpart")
# A wrapper function that extracts the predicted probabilities from rpart predictions:
mypred_rpart <- function(object, newdata) {
  predict(object, newdata, type="prob")[,2]
}
# Control parameters for rpart:
ctrl <- rpart.control(cp=0.003)

# Perform 5-repeated 5-fold non-spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                  model_fun = rpart, model_args = list(control=ctrl),
                  pred_fun = mypred_rpart, 
                  smp_fun = partition_cv, smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )


# Perform 5-repeated 5-fold SPATIAL cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                  model_fun = rpart, model_args = list(control=ctrl), 
                  pred_fun = mypred_rpart, 
                  smp_fun = partition_kmeans, smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )



####################################
# Random forest
####################################

library("randomForest")

# Exercise for you:
# Try using the random forest implementation in the ranger package instead!

# A wrapper function that extracts the predicted 
# probabilities from rpart predictions:
mypred_rf <- function(object, newdata) {
  predict(object, newdata, type="prob")[,2]
}

# Perform 5-repeated 5-fold non-spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                 model_fun = randomForest, 
                 pred_fun = mypred_rf, 
                 smp_fun = partition_cv, smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )

# Perform 5-repeated 5-fold spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("x","y"),
                  model_fun = randomForest, pred_fun = mypred_rf, 
                  smp_fun = partition_kmeans, smp_args = list(repetition=1:5, nfold=5))
summary(res$error_rep)[c("train_auroc","test_auroc"),1:2]
# Degree of overfitting (the more negative, the worse):
diff( summary(res$error_rep)[c("train_auroc","test_auroc"),"mean"] )
