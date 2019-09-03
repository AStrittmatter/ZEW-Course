
########################  Load Packages  ########################

# List of required packages
pkgs <- c('tidyverse','grf','corrplot','plotmo')

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
covariates <- c("mileage", "age_car_years", "other_car_owner", "co2_em", "euro_norm",
                "inspection", "bmw_320", "opel_astra", "mercedes_c", 
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

#desc <- fBasics::basicStats(df) %>% t() %>% as.data.frame() %>% 
#          select(Mean, Stdev, Minimum, Maximum, nobs)
#print(round(desc, digits=1))


########################  Correlation Matrix  ########################

#corr = cor(df)
#corrplot(corr, type = "upper", tl.col = "black")


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
covariates_obs <- as.matrix(cbind(df_obs[,c(2:ncol(df_obs))])) #,noise_obs))
covariates_hold_out <- as.matrix(cbind(df_hold_out[,c(2:ncol(df_hold_out))])) #,noise_hold_out))

print('The data is now ready for your first analysis!')



ncol(covariates_obs)
nrow(covariates_obs)
nrow(covariates_hold_out)

forest <- regression_forest(covariates_obs, first_price_obs,
                            sample.fraction = 0.5, min.node.size = 1,
                            mtry = floor(1/2*ncol(covariates_obs)),
                            num.trees = 500,
                            honesty = TRUE, honesty.fraction = 0.5)


fit_in <- predict(forest)$predictions
fit_out <- predict(forest, newdata =covariates_hold_out)$predictions


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



vars <- variable_importance(forest, decay.exponent = 0, max.depth =1)
rownames(vars) <- covariates
print(vars)






