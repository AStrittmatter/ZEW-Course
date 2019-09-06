
########################  Load Packages  ########################

# List of required packages
pkgs <- c('tidyverse','grf')

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
outcomes <- c("first_price", "final_price")

# Covariates/Features
covariates <- c("mileage", "age_car_years", "other_car_owner", "co2_em", "euro_norm", "inspection", 
                "bmw_320", "opel_astra", "mercedes_c", "vw_golf", "vw_passat", "diesel", 
                "private_seller", "guarantee", "maintenance_cert", "pm_green") 

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


########################  Take Hold-Out-Sample  ########################

df_part <- modelr::resample_partition(df, c(obs = 0.8, hold_out = 0.2))
df_obs <- as.data.frame(df_part$obs) # Training and estimation sample
df_hold_out <- as.data.frame(df_part$hold_out) # Hold-out-sample

# Outcomes
first_price_obs <- as.matrix(df_obs[,1])
first_price_hold_out <- as.matrix(df_hold_out[,1])
final_price_obs <- as.matrix(df_obs[,2])
final_price_hold_out <- as.matrix(df_hold_out[,2])

# Covariates/Features
covariates_obs <- as.matrix(df_obs[,c(3:ncol(df_obs))])
covariates_hold_out <- as.matrix(df_hold_out[,c(3:ncol(df_hold_out))])

print('The data is now ready for your first analysis!')


########################  Random Forest  ######################## 

forest <- regression_forest(covariates_obs, first_price_obs,
                            sample.fraction = 0.5, min.node.size = 1,
                            mtry = floor(1/2*ncol(covariates_obs)),
                            num.trees = 500,
                            honesty = TRUE, honesty.fraction = 0.5)


fit_in <- predict(forest)$predictions
fit_out <- predict(forest, newdata =covariates_hold_out)$predictions

print('Forest was build.')


# In-sample performance measures
mse1_in <- round(mean((first_price_obs - fit_in)^2),digits=3)
rsquared_in <- round(1-mean((first_price_obs - fit_in)^2)/mean((first_price_obs - mean(first_price_obs))^2),digits=3)
print(paste0("In-Sample MSE Forest: ", mse1_in))
print(paste0("In-Sample R-squared Forest: ", rsquared_in))

# Out-of-sample performance measures
mse1_out <- round(mean((first_price_hold_out - fit_out)^2),digits=3)
rsquared_out <- round(1-mean((first_price_hold_out - fit_out)^2)/mean((first_price_hold_out - mean(first_price_hold_out))^2),digits=3)
print(paste0("Out-of-Sample MSE Forest: ", mse1_out))
print(paste0("Out-of-Sample R-squared Forest: ", rsquared_out))

vars <- split_frequencies(forest, max.depth = 4)
colnames(vars) <- covariates
print(vars)

sum(vars[1,])
sum(vars[2,])
sum(vars[3,])
sum(vars[4,])

########################  Select tuning parameters for forest  ######################## 

#for_sizes = c(1,5, 10,15, 20,25,50,100,150,200,250, 300,350, 400, 450, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000)
for_sizes = c(1,5, 10,15, 20,25,50,100,150,200,250, 300,350, 400, 450, 500) # Because of computation time we consider only forests with 500 trees. At home you can consider larger forests.
auc <- matrix(NA,nrow=length(for_sizes),ncol=3)
ctr <- 0
for (n in for_sizes){
  ctr <- ctr + 1
  auc[ctr,1] <- n
  
    forest <- regression_forest(covariates_obs, first_price_obs,
                            sample.fraction = 0.5, min.node.size = 1,
                            mtry = floor(1/2*ncol(covariates_obs)),
                            num.trees = n,
                            honesty = TRUE, honesty.fraction = 0.5)
  
  # Predict prices in hold-out-sample
  pred_forest <- predict(forest, newdata = covariates_hold_out)
  rmse_forest <- round(sqrt(mean((first_price_hold_out - pred_forest$predictions)^2)),digits=3)
  auc[ctr,2] <- rmse_forest
  if (ctr >1) {
    auc[ctr,3] <- rmse_forest-auc[ctr-1,2]
  }
}

plot(auc[,1], auc[,2], main="Tuning of forest size", xlab="Number of trees in forest ", ylab="RMSE", pch=19)
nls_fit <- lm(auc[,2] ~  auc[,1] + I(auc[,1]^(1/2)) + I(auc[,1]^2) + I(auc[,1]^3) + I(log(auc[,1])))
lines(auc[,1], predict(nls_fit), col = "red")

plot(auc[c(2:nrow(auc)),1], auc[c(2:nrow(auc)),3], main="Tuning of forest size", xlab="Number of trees in forest ", ylab="Delta RMSE", pch=19)
nls_fit <- lm(auc[c(2:nrow(auc)),3] ~  auc[c(2:nrow(auc)),1] + I(auc[c(2:nrow(auc)),1]^(1/2)) + I(auc[c(2:nrow(auc)),1]^2) + I(auc[c(2:nrow(auc)),1]^3) + I(log(auc[c(2:nrow(auc)),1])))
lines(auc[c(2:nrow(auc)),1], predict(nls_fit), col = "red")
abline(h=0)





########################  Random Forest for "Overpice"  ######################## 

forest <- regression_forest(covariates_obs, final_price_obs,
                            sample.fraction = 0.5, min.node.size = 1,
                            mtry = floor(1/2*ncol(covariates_obs)),
                            num.trees = 500,
                            honesty = TRUE, honesty.fraction = 0.5)


fit_in <- predict(forest)$predictions
fit_out <- predict(forest, newdata =covariates_hold_out)$predictions

# In-sample performance measures
mse2_in <- round(mean((final_price_obs - fit_in)^2),digits=3)
rsquared2_in <- round(1-mean((final_price_obs - fit_in)^2)/mean((final_price_obs - mean(final_price_obs))^2),digits=3)
print(paste0("In-Sample MSE Forest: ", mse2_in))
print(paste0("In-Sample R-squared Forest: ", rsquared2_in))

# Out-of-sample performance measures
mse2_out <- round(mean((final_price_hold_out - fit_out)^2),digits=3)
rsquared2_out <- round(1-mean((final_price_hold_out - fit_out)^2)/mean((final_price_hold_out - mean(final_price_hold_out))^2),digits=3)
print(paste0("Out-of-Sample MSE Forest: ", mse2_out))
print(paste0("Out-of-Sample R-squared Forest: ", rsquared2_out))


