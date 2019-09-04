
########################  Load Packages  ########################

# List of required packages
pkgs <- c('tidyverse','glmnet','corrplot','plotmo')

# Load packages
for(pkg in pkgs){
    library(pkg, character.only = TRUE)
}

set.seed(10101) # set starting value for random number generator

print('All packages successfully installed and loaded.')


########################  Load Data Frame  ########################

# Load data frame
data_raw <- read.csv("Data/used_cars.csv",header=TRUE, sep=",")

# Outcome Variable
outcomes <- c("first_price")

# Covariates/Features
covariates <- c("mileage","mileage2", "mileage3", "mileage4", "age_car_years", "age_car_years2", 
                "age_car_years3", "age_car_years4", "other_car_owner", "co2_em", "euro_1", 
                "euro_2", "euro_3", "euro_4", "euro_6", "dur_next_ins_0", 
                "dur_next_ins_1_2", "bmw_320", "opel_astra", "mercedes_c", 
                "vw_passat", "diesel", "private_seller", "guarantee", 
                "maintenance_cert", "pm_green") 

all_variables <- c(outcomes, covariates)

# Selection of Subsample size, max. 104,721 observations
# Select smaller subsample to decrease computation time
n_obs <- 300
df <- data_raw %>%
  dplyr::sample_n(n_obs) %>%
  dplyr::select(all_variables)

print('Data frame successfully loaded.')


########################  Table with Descriptive Statistics  ########################

desc <- fBasics::basicStats(df) %>% t() %>% as.data.frame() %>% 
          select(Mean, Stdev, Minimum, Maximum, nobs)
print(round(desc, digits=1))


########################  Correlation Matrix  ########################

corr = cor(df)
corrplot(corr, type = "upper", tl.col = "black")


########################  Take Hold-Out-Sample  ########################

df_part <- modelr::resample_partition(df, c(obs = 0.8, hold_out = 0.2))
df_obs <- as.data.frame(df_part$obs) # Training and estimation sample
df_hold_out <- as.data.frame(df_part$hold_out) # Hold-out-sample

# Outcomes
first_price_obs <- as.matrix(df_obs[,1])
first_price_hold_out <- as.matrix(df_hold_out[,1])

# Generate some noisy covariates to disturbe the estimation
noise <- c("noise1", "noise2", "noise3", "noise4", "noise5", "noise6", "noise7", "noise8", "noise9", "noise10", 
           "noise11", "noise12", "noise13", "noise14", "noise15", "noise16", "noise17", "noise18", "noise19", "noise20", 
           "noise21", "noise22", "noise23", "noise24", "noise25", "noise26", "noise27", "noise28", "noise29", "noise30", 
           "noise31", "noise32", "noise33", "noise34", "noise35", "noise36", "noise37", "noise38", "noise39", "noise40",
           "noise41", "noise42", "noise43", "noise44", "noise45", "noise46", "noise47", "noise48", "noise49", "noise50", 
           "noise51", "noise52", "noise53", "noise54", "noise55", "noise56", "noise57", "noise58", "noise59", "noise60", 
           "noise61", "noise62", "noise63", "noise64", "noise65", "noise66", "noise67", "noise68", "noise69", "noise70", 
           "noise71", "noise72", "noise73", "noise74", "noise75", "noise76", "noise77", "noise78", "noise79", "noise80", 
           "noise81", "noise82", "noise83", "noise84", "noise85", "noise86", "noise87", "noise88", "noise89", "noise90", 
           "noise91", "noise92", "noise93", "noise94", "noise95", "noise96", "noise97", "noise98", "noise99", "noise100")
noise_obs <- matrix(data = rnorm(nrow(df_obs)*100),  nrow = nrow(df_obs), ncol = 100)
colnames(noise_obs) <- noise
noise_hold_out <- matrix(data = rnorm(nrow(df_hold_out)*100),  nrow = nrow(df_hold_out), ncol = 100)
colnames(noise_hold_out) <- noise
covariates <- c(covariates, noise)

# Covariates/Features
covariates_obs <- as.matrix(cbind(df_obs[,c(2:ncol(df_obs))],noise_obs))
covariates_hold_out <- as.matrix(cbind(df_hold_out[,c(2:ncol(df_hold_out))],noise_hold_out))

print('The data is now ready for your first analysis!')


########################  OLS Model  ######################## 

# Setup the formula of the linear regression model
sumx <- paste(covariates, collapse = " + ")  
linear <- paste("first_price_obs",paste(sumx, sep=" + "), sep=" ~ ")
linear <- as.formula(linear)

# Setup the data for linear regression
data <- as.data.frame(cbind(first_price_obs,covariates_obs))

# Estimate OLS model
ols <- lm(linear, data)
# Some variables might be dropped because of perfect colinearity (121 covariates - 240 observations)

# In-sample fitted values
fit1_in <- predict.lm(ols)


summary(ols)

# Out-of-sample fitted values
fit1_out <- predict.lm(ols, newdata = data.frame(covariates_hold_out))


# In-sample performance measures
mse1_in <- round(mean((first_price_obs - fit1_in)^2),digits=3)
rsquared_in <- round(1-mean((first_price_obs - fit1_in)^2)/mean((first_price_obs - mean(first_price_obs))^2),digits=3)
print(paste0("In-Sample MSE OLS: ", mse1_in))
print(paste0("In-Sample R-squared OLS: ", rsquared_in))

# Out-of-sample performance measures
mse1_out <- round(mean((first_price_hold_out - fit1_out)^2),digits=3)
rsquared_out <- round(1-mean((first_price_hold_out - fit1_out)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)
print(paste0("Out-of-Sample MSE OLS: ", mse1_out))
print(paste0("Out-of-Sample R-squared OLS: ", rsquared_out))


########################  CV-LASSO  ######################## 
p = 1 # 1 for LASSO, 0 for Ridge

set.seed(10101)
lasso.linear <- cv.glmnet(covariates_obs, first_price_obs, alpha=p, 
                          nlambda = 100, type.measure = 'mse')
# nlambda specifies the number of different lambda values on the grid (log-scale)
# type.measure spciefies that the optimality criteria is the MSE in CV-samples

# Plot MSE in CV-Samples for different values of lambda
plot(lasso.linear)

# Optimal Lambda
print(paste0("Lambda minimising CV-MSE: ", round(lasso.linear$lambda.min,digits=8)))
# 1 standard error rule reduces the number of included covariates
print(paste0("Lambda 1 standard error rule: ", round(lasso.linear$lambda.1se,digits=8)))

# Number of Non-Zero Coefficients
print(paste0("Number of selected covariates (lambd.min): ",lasso.linear$glmnet.fit$df[lasso.linear$glmnet.fit$lambda==lasso.linear$lambda.min]))
print(paste0("Number of selected covariates (lambd.1se): ",lasso.linear$glmnet.fit$df[lasso.linear$glmnet.fit$lambda==lasso.linear$lambda.1se]))


########################  Visualisation of LASSO  ######################## 

glmcoef<-coef(lasso.linear,lasso.linear$lambda.1se)
coef.increase<-dimnames(glmcoef[glmcoef[,1]>0,0])[[1]]
coef.decrease<-dimnames(glmcoef[glmcoef[,1]<0,0])[[1]]

lambda_min =  lasso.linear$glmnet.fit$lambda[29]/lasso.linear$glmnet.fit$lambda[1]
set.seed(10101)
mod <- glmnet(covariates_obs, first_price_obs, lambda.min.ratio = lambda_min, alpha=p)
maxcoef<-coef(mod,s=lambda_min)
coef<-dimnames(maxcoef[maxcoef[,1]!=0,0])[[1]]
allnames<-dimnames(maxcoef[maxcoef[,1]!=0,0])[[1]][order(maxcoef[maxcoef[,1]!=0,ncol(maxcoef)],decreasing=TRUE)]
allnames<-setdiff(allnames,allnames[grep("Intercept",allnames)])

#assign colors
cols<-rep("gray",length(allnames))
cols[allnames %in% coef.increase]<-"red"   
cols[allnames %in% coef.decrease]<- "green"

plot_glmnet(mod,label=TRUE,s=lasso.linear$lambda.1se,col= cols)


########################  Plot LASSO Coefficients  ########################

print('LASSO coefficients')

glmcoef<-coef(lasso.linear, lasso.linear$lambda.1se)
print(glmcoef)
# the LASSO coefficients are biased because of the penalty term


######################## In-Sample Performance of LASSO  ######################## 

# Estimate LASSO model 
# Use Lambda that minizes CV-MSE
set.seed(10101)
lasso.fit.min <- glmnet(covariates_obs, first_price_obs, lambda = lasso.linear$lambda.min)
yhat.lasso.min <- predict(lasso.fit.min, covariates_obs)

# Use 1 standard error rule
set.seed(10101)
lasso.fit.1se <- glmnet(covariates_obs, first_price_obs, lambda = lasso.linear$lambda.1se)
yhat.lasso.1se <- predict(lasso.fit.1se, covariates_obs)

# In-sample performance measures
print(paste0("In-Sample MSE OLS: ", mse1_in))
print(paste0("In-Sample R-squared OLS: ", rsquared_in))

mse2_in <- round(mean((first_price_obs - yhat.lasso.min)^2),digits=3)
rsquared2_in <- round(1-mean((first_price_obs - yhat.lasso.min)^2)/mean((first_price_obs - mean(first_price_obs))^2),digits=3)
print(paste0("In-Sample MSE Lasso (lambda.min): ", mse2_in))
print(paste0("In-Sample R-squared Lasso (lambda.min): ", rsquared2_in))

mse3_in <- round(mean((first_price_obs - yhat.lasso.1se)^2),digits=3)
rsquared3_in <- round(1-mean((first_price_obs - yhat.lasso.1se)^2)/mean((first_price_obs - mean(first_price_obs))^2),digits=3)
print(paste0("In-Sample MSE Lasso(lambda.1se): ", mse3_in))
print(paste0("In-Sample R-squared Lasso (lambda.1se): ", rsquared3_in))


######################## Out-of-Sample Performance of LASSO  ######################## 

# Extrapolate Lasso fitted values to hold-out-sample
yhat.lasso.min <- predict(lasso.fit.min, covariates_hold_out)
yhat.lasso.1se <- predict(lasso.fit.1se, covariates_hold_out)

# Out-of-sample performance measures
print(paste0("Out-of-Sample MSE OLS: ", mse1_out))
print(paste0("Out-of-Sample R-squared OLS: ", rsquared_out))

mse2_out <- round(mean((first_price_hold_out - yhat.lasso.min)^2),digits=3)
rsquared2_out <- round(1-mean((first_price_hold_out - yhat.lasso.min)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)
print(paste0("Out-of-Sample MSE Lasso (lambda.min): ", mse2_out))
print(paste0("Out-of-Sample R-squared Lasso (lambda.min): ", rsquared2_out))

mse3_out <- round(mean((first_price_hold_out - yhat.lasso.1se)^2),digits=3)
rsquared3_out <- round(1-mean((first_price_hold_out - yhat.lasso.1se)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)
print(paste0("Out-of-Sample MSE Lassso (lambda.1se): ", mse3_out))
print(paste0("Out-of-Sample R-squared Lasso (lambda.1se): ", rsquared3_out))


######################## Separate Training and Estimation Sample  ######################## 

df_obs2 <- as.data.frame(cbind(first_price_obs,covariates_obs))
df_part2 <-  modelr::resample_partition(df_obs2, c(obs = 0.5, hold_out = 0.5))
df_train <- as.data.frame(df_part2$obs) # Training sample
df_est <- as.data.frame(df_part2$hold_out) # Estimation sample

# Outcomes
first_price_train <- as.matrix(df_train[,1])
first_price_est <- as.matrix(df_est[,1])

# Covariates/Features
covariates_train <- as.matrix(df_train[,c(2:ncol(df_obs2))])
covariates_est <- as.matrix(df_est[,c(2:ncol(df_obs2))])


# Crossvalidate Lasso model
set.seed(10101)
lasso.linear2 <- cv.glmnet(covariates_train, first_price_train, alpha=p, 
                          nlambda = 100, type.measure = 'mse')
plot(lasso.linear2)

# Optimal Lambda
# 1 standard error rule reduces the number of included covariates
print(paste0("Lambda 1 standard error rule: ", round(lasso.linear2$lambda.1se,digits=8)))

# Number of Non-Zero Coefficients
print(paste0("Number of selected covariates (lambd.1se): ",lasso.linear2$glmnet.fit$df[lasso.linear$glmnet.fit$lambda==lasso.linear$lambda.1se]))

# Estimate LASSO model 
set.seed(10101)
lasso.fit2.1se <- glmnet(covariates_est, first_price_est, lambda = lasso.linear2$lambda.1se)
yhat2.lasso.1se <- predict(lasso.fit2.1se, covariates_hold_out)


# Out-of-sample performance measures
mse4_out <- round(mean((first_price_hold_out - yhat2.lasso.1se)^2),digits=3)
rsquared4_out <- round(1-mean((first_price_hold_out - yhat2.lasso.1se)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)

print(paste0("Out-of-Sample MSE OLS: ", mse1_out))
print(paste0("Out-of-Sample R-squared OLS: ", rsquared_out))

print(paste0("Out-of-Sample MSE Lassso (lambda.1se): ", mse3_out))
print(paste0("Out-of-Sample R-squared Lasso (lambda.1se): ", rsquared3_out))

print(paste0("Out-of-Sample MSE Honest Lassso (lambda.1se): ", mse4_out))
print(paste0("Out-of-Sample R-squared Honest Lasso (lambda.1se): ", rsquared4_out))


######################## Cross-Fitting  ######################## 

# Crossvalidate Lasso model
set.seed(10101)
lasso.linear3 <- cv.glmnet(covariates_est, first_price_est, alpha=p, 
                          nlambda = 100, type.measure = 'mse')
plot(lasso.linear3)

# Optimal Lambda
# 1 standard error rule reduces the number of included covariates
print(paste0("Lambda 1 standard error rule: ", round(lasso.linear3$lambda.1se,digits=8)))

# Number of Non-Zero Coefficients
print(paste0("Number of selected covariates (lambd.1se): ",lasso.linear3$glmnet.fit$df[lasso.linear$glmnet.fit$lambda==lasso.linear$lambda.1se]))

# Estimate LASSO model 
set.seed(10101)
lasso.fit3.1se <- glmnet(covariates_train, first_price_train, lambda = lasso.linear3$lambda.1se)
yhat3.lasso.1se_B <- predict(lasso.fit3.1se, covariates_hold_out)


# Take average of fitted values from both cross-fitting samples
yhat3.lasso.1se <- 0.5*(yhat2.lasso.1se + yhat3.lasso.1se_B)


# Out-of-sample performance measures
mse5_out <- round(mean((first_price_hold_out - yhat3.lasso.1se)^2),digits=3)
rsquared5_out <- round(1-mean((first_price_hold_out - yhat3.lasso.1se)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)

print(paste0("Out-of-Sample MSE OLS: ", mse1_out))
print(paste0("Out-of-Sample R-squared OLS: ", rsquared_out))

print(paste0("Out-of-Sample MSE Lassso (lambda.1se): ", mse3_out))
print(paste0("Out-of-Sample R-squared Lasso (lambda.1se): ", rsquared3_out))

print(paste0("Out-of-Sample MSE Honest Lassso (lambda.1se): ", mse4_out))
print(paste0("Out-of-Sample R-squared Honest Lasso (lambda.1se): ", rsquared4_out))

print(paste0("Out-of-Sample MSE Cross-Fitted Honest Lassso (lambda.1se): ", mse5_out))
print(paste0("Out-of-Sample R-squared Cross-Fitted Honest Lasso (lambda.1se): ", rsquared5_out))



