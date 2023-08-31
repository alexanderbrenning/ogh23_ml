#####################################################
# OpenGeoHub Summer School 2023, Poznan
# Alexander Brenning
# University of Jena, Germany
#####################################################
# Model interpretation
# Case study: Mapping crop types from Landsat imagery
#####################################################

# Fewer decimal places - works better for instructor:
options(digits=4)

library("sperrorest")    # PVI in spatial CV
library("randomForest")  # Random forest
library("rpart")         # CART
library("pdp")           # Partial dependence plots
library("ggplot2")       # Plotting

# Load the data set:
d <- readRDS("Maipo_fields.rds")

# add this many random noise variable to the data set:
p_noise <- 10 

# Add 10 noise variables for illustrative purposes:
for(i in 0:p_noise){
  d[paste('rdmvar', i, sep = "")] <- rnorm(nrow(d))
}

#### Choose one of the following formula objects:

fo1 <- croptype ~ ndvi01 + ndvi02 + ndvi03 + ndvi04 + ndvi05 +
 ndvi06 + ndvi07 + ndvi08 + rdmvar1

# Use a larger feature set than previously,
# including the 'noise' variables:
fo2 <- as.formula(
  paste("croptype ~",
        paste( paste("b", outer(1:8,2:7,paste,sep = ""), sep=""), 
               collapse="+" ), "+",
        paste( paste("ndvi0", 1:8, sep = ""), collapse="+"), "+",
        paste( paste("ndwi0", 1:8, sep = ""), collapse="+"), "+",
        paste( paste("rdmvar", 0:p_noise, sep = ""), collapse="+")
  ))
# What have we done:
fo2

# Pick one of fo1, fo2:
fo <- fo1


############################################################
### Random Forest:
### start with tools from the randomForest package
############################################################

fit <- randomForest(fo, data = d) # try e.g. nodesize = 60

randomForest::varImpPlot(fit)
# type = 1 (mean decrease in accuracy) won't work with croptype data set??

# Examples of a PDP using function from randomForest package:
d2 <- d
d2$croptype <- NULL
par(mfrow = c(2,2))
randomForest::partialPlot(fit, x.var = "ndvi04", pred.data = d2, which.class = "crop1")
randomForest::partialPlot(fit, x.var = "ndvi04", pred.data = d2, which.class = "crop2")
randomForest::partialPlot(fit, x.var = "ndvi04", pred.data = d2, which.class = "crop3")
randomForest::partialPlot(fit, x.var = "ndvi04", pred.data = d2, which.class = "crop4")

rf_imp <- randomForest::importance(fit)
top3 <- names(rf_imp[order(rf_imp[,"MeanDecreaseGini"], decreasing = TRUE),])[1:3]
partialPlot(fit, x.var = top3[1], pred.data = d, which.class = "crop1", xlab = top3[1])
partialPlot(fit, x.var = top3[2], pred.data = d, which.class = "crop1", xlab = top3[2])
partialPlot(fit, x.var = top3[3], pred.data = d, which.class = "crop1", xlab = top3[3])



############################################################
### Random Forest:
### Use sperrorest for (spatial) variable importance
############################################################

# Calculate importance for these variables:
imp_vars <- all.vars(fo)[-1]
imp_vars

# Perform spatial cross-validation; using simplified settings to test the code:
res <- sperrorest(formula = fo, data = d, coords = c("utmx","utmy"),
                  model_fun = randomForest, 
                  model_args = list(ntree = 100), 
                  pred_args = list(type="class"),
                  smp_fun = partition_kmeans, smp_args = list(repetition=1:5, nfold=10),
                  importance = TRUE, imp_permutations = 10,
                  imp_variables = imp_vars)
# Cross-validation estimate of error rate & accuracy:
imp <- summary(res$importance)
# ... a data.frame with results...

# E.g. mean decrease in accuracy when permuting ndvi01:
imp["ndvi04", "mean.accuracy"]
# Its standard deviation over the repetitions:
imp["ndvi04", "sd.accuracy"]

imp <- imp[order(imp$mean.accuracy, decreasing = TRUE),]
imp[1:5, c("mean.accuracy", "sd.accuracy")]

# Create a barplot - looks better with greater importances at the top:
imp <- imp[order(imp$mean.accuracy, decreasing = FALSE),]
imp[1:5, c("mean.accuracy", "sd.accuracy")]
par(mar = c(5,7,1,1)) # bottom, left, top, right margins
barplot(imp$mean.accuracy, names.arg = rownames(imp), horiz = TRUE, las = 1, 
        xlab = "Mean decrease in accuracy")



############################################################
### Random Forest:
### use pdp package for partial dependence plots
############################################################

# Let's just focus on these three predictors for now:
top3 <- rownames(imp[order(imp$mean.accuracy, decreasing = TRUE),])[1:3]
# carefully read the ?partial help page! make sure you understand the settings!
autoplot(pdp::partial(fit, pred.var = top3[1], which.class = "crop1"))
autoplot(pdp::partial(fit, pred.var = top3[2], which.class = "crop1"))
autoplot(pdp::partial(fit, pred.var = top3[3], which.class = "crop1"))

# ...only looking at classification of crop1!


############################################################
### Random Forest:
### use iml package for partial dependence and ALE plot
############################################################

library("iml")        # ML model interpretation
library("ggplot2")    # Plotting
library("patchwork")  # Plotting

# use the smaller feature set, fo1:
fo <- fo1

fit <- randomForest(fo, data = d, importance = TRUE)

predvars <- all.vars(fo)[-1]
predictors <- d[, predvars]
response <- d$croptype

predictor <- iml::Predictor$new(fit, data = predictors, y = response)

# Permutation variable importance on the training set (!):
imp <- iml::FeatureImp$new(predictor, loss = "ce", n.repetitions = 100, compare = "difference")
plot(imp)

# PVI, randomForest style (for comparison)
par(mfrow = c(1,1))
varImpPlot(fit, type = 1)


# Calculate SHAP feature importance from Shapley values:
# This is very SLOW and not parallelized!
extract_shapleys <- function(x) 
  Shapley$new(predictor, 
              x.interest = predictors[x,],
              sample.size = 10 # use default of 100! -> slower
             )$results$phi
shaps <- sapply(1:nrow(d), extract_shapleys)
shap <- apply(abs(shaps), 1, mean)
# average over all four classes:
shap <- tapply(shap, rep(predvars, 4), mean)
barplot(shap[order(shap)], horiz = TRUE, las = 1, xlab = "SHAP feature importance")




# Interaction strength using iml package:
interac <- iml::Interaction$new(predictor)
plot(interac)

# ndvi01, ndvi03 and ndvi04 showed the strongest interactions with other
# predictors, therefore take a closer look at how they interact:

interac <- Interaction$new(predictor, feature = "ndvi01")
plot(interac)

interac <- Interaction$new(predictor, feature = "ndvi03")
plot(interac)

interac <- Interaction$new(predictor, feature = "ndvi04")
plot(interac)

# PDP / ALE plot using iml package (use method argument!):
effs <- iml::FeatureEffects$new(predictor, method = "pdp", features = top3)
plot(effs)

effs <- iml::FeatureEffects$new(predictor, method = "ale", features = top3)
plot(effs)





############################################################
### Repeat the above for classification trees
############################################################

fit <- rpart(fo, data = d, control = rpart.control(cp=0, maxdepth=5))
par(xpd=TRUE, mfrow=c(1,1))
plot(fit)
text(fit, use.n = TRUE)
par(xpd = FALSE)

# Control parameters for rpart:
ctrl <- rpart.control(cp = 0, maxdepth = 5)

# Calculate importance for these variables:
imp_vars <- all.vars(fo)[-1]
imp_vars

# Perform 5-repeated 10-fold spatial cross-validation:
res <- sperrorest(formula = fo, data = d, coords = c("utmx","utmy"),
                  model_fun = rpart, model_args = list(control=ctrl), 
                  pred_args = list(type="class"),
                  smp_fun = partition_kmeans, smp_args = list(repetition=1:5, nfold=10),
                  importance = TRUE, imp_permutations = 10,
                  imp_variables = imp_vars)
# Cross-validation estimate of AUROC:
imp <- summary(res$importance)
# ... a data.frame with results...

# E.g. mean decrease in accuracy when permuting ndvi01:
imp["ndvi05", "mean.accuracy"]
# Its standard deviation over the repetitions:
imp["ndvi05", "sd.accuracy"]

imp <- imp[order(imp$mean.accuracy, decreasing = TRUE),]
imp[1:5, c("mean.accuracy", "sd.accuracy")]

### Variable importance plot:

# Create a barplot - looks better with greater importances at the top:
imp <- imp[order(imp$mean.accuracy, decreasing = FALSE),]
par(mar = c(5,7,1,1)) # bottom, left, top, right margins
barplot(imp$mean.accuracy, names.arg = rownames(imp), horiz = TRUE, las = 1, 
        xlab = "Mean decrease in accuracy")

### Partial dependence plots for the top-ranking predictors:

top3 <- rownames(imp[order(imp$mean.accuracy, decreasing = TRUE),])[1:3]
# carefully read the ?partial help page! make sure you understand the settings!
autoplot(partial(fit, pred.var = top3[1], which.class = "crop1"))
autoplot(partial(fit, pred.var = top3[2], which.class = "crop1"))
autoplot(partial(fit, pred.var = top3[3], which.class = "crop1"))

# Do these plots accurately depict the classifier's structure? (see tree plot!)
# What is missing?

# Hint: Does it get better if we look at the following subsets:
autoplot(partial(fit, pred.var = "ndvi04", which.class = "crop1",
                 train = d[d$ndvi01 < 0.4,]))
autoplot(partial(fit, pred.var = "ndvi04", which.class = "crop1",
                 train = d[d$ndvi01 >= 0.4,]))
# ...compared to:
autoplot(partial(fit, pred.var = "ndvi04", which.class = "crop1"))
