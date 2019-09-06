
########################  Load Packages  ########################

# List of required packages
pkgs <- c('corrplot', 'glmnet', 'tidyverse')

# Load packages
for(pkg in pkgs){
    library(pkg, character.only = TRUE)
}

set.seed(10101) # set starting value for random number generator

print('All packages successfully installed and loaded.')

########################  Load Data Frame  ########################
setwd('C:/Users/user/Dropbox/ZEW_Vorlesung/3_PC/3_Exercise/')
# Load data frame
df <- read.csv("Data/job_corps.csv",header=TRUE, sep=",")

# Outcome Variable
outcome <- c("EARNY4")

# Treatment Variables
treatment <- c("assignment", "participation")

# Covariates/Features
covariates <- c("female", "age_1", "age_2", "age_3", "ed0_6", "ed6_12", "hs_ged", "white", "black", "hisp",
                "oth_eth", "haschld", "livespou", "everwork", "yr_work", "currjob", "job0_3", "job3_9", "job9_12",
                "earn1", "earn2", "earn3", "earn4", "badhlth", "welf_kid", "got_fs", "publich", "got_afdc",
                "harduse", "potuse", "evarrst", "pmsa", "msa")
    
all_variables <- c(outcome, treatment, covariates)

print('Data frame successfully loaded and sample selected.')

########################  Table with Descriptive Statistics  ########################
desc <- fBasics::basicStats(df) %>% t() %>% as.data.frame() %>% 
  select(Mean, Stdev, Minimum, Maximum, nobs)
print(round(desc, digits=2))

# Print as tex-file
#kable(desc, "latex", booktabs = T)

########################  Correlation Matrix  ########################

corr = cor(df)
corrplot(corr, type = "upper", tl.col = "black")

# Save correlation matrix as png-file
png(height=1200, width=1200, file="correlation.png")
    corrplot(corr, type = "upper", tl.col = "black")
dev.off()

########################  Extract Dataset  ########################
set.seed(10101)

# Extracting Data 
data <- df %>% dplyr::select(all_variables)

# Setting up the data, renaming columns and discarding rows with NA (if any)
df <- bind_cols(data) %>%  na.omit()

# Add First Order Interactions
interactions <-model.matrix(~ (female + age_1 + age_2 + age_3 + ed0_6 + ed6_12 + hs_ged + white + black + hisp
                 + oth_eth + haschld + livespou + everwork + yr_work + currjob + job0_3 + job3_9 + job9_12
                 + earn1 + earn2 + earn3 + earn4 + badhlth + welf_kid + got_fs + publich + got_afdc
                 + harduse + potuse + evarrst + pmsa + msa)^2,df)
df <- cbind(df,interactions[,c(35:ncol(interactions))])

print('Data successfully extracted.')

########################  Partition the Samples  ########################

# Partition Hold-Out-Sample
df_part <- modelr::resample_partition(df, c(obs = 0.9, hold_out = 0.1))
df_obs <- as.data.frame(df_part$obs) # Training and estimation sample
df_hold_out <- as.data.frame(df_part$hold_out) # Hold-out-sample

# Partition Samples for Cross-Fitting
df_part <- modelr::resample_partition(df_obs, c(obs_A = 0.5, obs_B = 0.5))
df_obs_A <- as.data.frame(df_part$obs_A) # Sample A
df_obs_B <- as.data.frame(df_part$obs_B) # Sample B

# Partition Samples A and B for Honest Inference
df_part <- modelr::resample_partition(df_obs_A, c(obs_A_train = 0.5, obs_A_est = 0.5))
df_obs_A_train <- as.data.frame(df_part$obs_A_train) # Training Sample A
df_obs_A_est <- as.data.frame(df_part$obs_A_est) # Estimation Sample A
df_part <- modelr::resample_partition(df_obs_B, c(obs_B_train = 0.5, obs_B_est = 0.5))
df_obs_B_train <- as.data.frame(df_part$obs_B_train) # Training Sample B
df_obs_B_est <- as.data.frame(df_part$obs_B_est) # Estimation Sample B

print('Samples are partitioned.')

########################  Generate Variables  ########################

# Outcome
earnings_hold_out <- as.matrix(df_hold_out[,1])
earnings_obs <- as.matrix(df_obs[,1])
earnings_obs_A <- as.matrix(df_obs_A[,1])
earnings_obs_B <- as.matrix(df_obs_B[,1])
earnings_obs_A_train <- as.matrix(df_obs_A_train[,1])
earnings_obs_A_est <- as.matrix(df_obs_A_est[,1])
earnings_obs_B_train <- as.matrix(df_obs_B_train[,1])
earnings_obs_B_est <- as.matrix(df_obs_B_est[,1])

# Treatment
treat = 2 #Select treatmen 2= offer to participate, 3 = actual participation
treat_hold_out <- as.matrix(df_hold_out[,treat])
treat_obs <- as.matrix(df_obs[,treat])
treat_obs_A <- as.matrix(df_obs_A[,treat])
treat_obs_B <- as.matrix(df_obs_B[,treat])
treat_obs_A_train <- as.matrix(df_obs_A_train[,treat])
treat_obs_A_est <- as.matrix(df_obs_A_est[,treat])
treat_obs_B_train <- as.matrix(df_obs_B_train[,treat])
treat_obs_B_est <- as.matrix(df_obs_B_est[,treat])

# Covariates
covariates_hold_out <- as.matrix(df_hold_out[,c(4:ncol(df_hold_out))])
covariates_obs <- as.matrix(df_obs[,c(4:ncol(df_obs))])
covariates_obs_A <- as.matrix(df_obs_A[,c(4:ncol(df_obs_A))])
covariates_obs_B <- as.matrix(df_obs_B[,c(4:ncol(df_obs_B))])
covariates_obs_A_train <- as.matrix(df_obs_A_train[,c(4:ncol(df_obs_A_train))])
covariates_obs_A_est <- as.matrix(df_obs_A_est[,c(4:ncol(df_obs_A_est))])
covariates_obs_B_train <- as.matrix(df_obs_B_train[,c(4:ncol(df_obs_B_train))])
covariates_obs_B_est <- as.matrix(df_obs_B_est[,c(4:ncol(df_obs_B_est))])

print('The data is now ready for your analysis!')

########################  Conditional Potential Earnings  ########################
p = 1 # 1 for LASSO, 0 for Ridge
set.seed(10101)

## Using Sample A to Predict Sample B
# Potential Earnings under Treatment
lasso_y1_A_train <- cv.glmnet(covariates_obs_A_train[treat_obs_A_train==1,], earnings_obs_A_train[treat_obs_A_train==1,],
                              alpha=p, type.measure = 'mse', lambda.min.ratio = 0.01, nlambda = 50, parallel=FALSE)
plot(lasso_y1_A_train)
fit_y1_A_est <- glmnet(covariates_obs_A_est[treat_obs_A_est==1,], earnings_obs_A_est[treat_obs_A_est==1,]
                        ,lambda = lasso_y1_A_train$lambda.min)
y1hat_B <- predict(fit_y1_A_est, covariates_obs_B, type = 'response')
y1hat_B_train <- predict(fit_y1_A_est, covariates_obs_B_train, type = 'response')
y1hat_B_est <- predict(fit_y1_A_est, covariates_obs_B_est, type = 'response')

# Potential Earnings under Non-Treatment
lasso_y0_A_train <- cv.glmnet(covariates_obs_A_train[treat_obs_A_train==0,], earnings_obs_A_train[treat_obs_A_train==0,],
                              alpha=p, type.measure = 'mse', lambda.min.ratio = 0.01, nlambda = 50, parallel=FALSE)
plot(lasso_y0_A_train)
fit_y0_A_est <- glmnet(covariates_obs_A_est[treat_obs_A_est==0,], earnings_obs_A_est[treat_obs_A_est==0,]
                        ,lambda = lasso_y0_A_train$lambda.min)
y0hat_B <- predict(fit_y0_A_est, covariates_obs_B, type = 'response')
y0hat_B_train <- predict(fit_y0_A_est, covariates_obs_B_train, type = 'response')
y0hat_B_est <- predict(fit_y0_A_est, covariates_obs_B_est, type = 'response')

## Using Sample B to Predict Sample A
# Potential Earnings under Treatment
lasso_y1_B_train <- cv.glmnet(covariates_obs_B_train[treat_obs_B_train==1,], earnings_obs_B_train[treat_obs_B_train==1,],
                              alpha=p, type.measure = 'mse', lambda.min.ratio = 0.01, nlambda = 50, parallel=FALSE)
plot(lasso_y1_B_train)
fit_y1_B_est <- glmnet(covariates_obs_B_est[treat_obs_B_est==1,], earnings_obs_B_est[treat_obs_B_est==1,]
                        ,lambda = lasso_y1_B_train$lambda.min)
y1hat_A <- predict(fit_y1_B_est, covariates_obs_A, type = 'response')
y1hat_A_train <- predict(fit_y1_B_est, covariates_obs_A_train, type = 'response')
y1hat_A_est <- predict(fit_y1_B_est, covariates_obs_A_est, type = 'response')

# Potential Earnings under Non-Treatment
lasso_y0_B_train <- cv.glmnet(covariates_obs_B_train[treat_obs_B_train==0,], earnings_obs_B_train[treat_obs_B_train==0,],
                              alpha=p, type.measure = 'mse', lambda.min.ratio = 0.01, nlambda = 50, parallel=FALSE)
plot(lasso_y0_B_train)
fit_y0_B_est <- glmnet(covariates_obs_B_est[treat_obs_B_est==0,], earnings_obs_B_est[treat_obs_B_est==0,]
                        ,lambda = lasso_y0_B_train$lambda.min)
y0hat_A <- predict(fit_y0_B_est, covariates_obs_A, type = 'response')
y0hat_A_train <- predict(fit_y0_B_est, covariates_obs_A_train, type = 'response')
y0hat_A_est <- predict(fit_y0_B_est, covariates_obs_A_est, type = 'response')

########################  Propensity Score  ########################
p = 1 # 1 for LASSO, 0 for Ridge
set.seed(10101)

# Using Sample A to Predict Sample B
lasso_p_A_train <- cv.glmnet(covariates_obs_A_train, treat_obs_A_train, alpha=p, type.measure = 'mse',
                             lambda.min.ratio = 0.1, nlambda = 50, family='binomial', parallel=FALSE)
plot(lasso_p_A_train)
fit_p_A_est <- glmnet(covariates_obs_A_est, treat_obs_A_est,lambda = lasso_p_A_train$lambda.min, family='binomial')
pscore_B <- predict(fit_p_A_est, covariates_obs_B, type = 'response')
pscore_B_train <- predict(fit_p_A_est, covariates_obs_B_train, type = 'response')
pscore_B_est <- predict(fit_p_A_est, covariates_obs_B_est, type = 'response')

# Using Sample B to Predict Sample A
lasso_p_B_train <- cv.glmnet(covariates_obs_B_train, treat_obs_B_train, alpha=p, type.measure = 'mse',
                             lambda.min.ratio = 0.1, nlambda = 50, family='binomial', parallel=FALSE)
plot(lasso_p_B_train)
fit_p_B_est <- glmnet(covariates_obs_B_est, treat_obs_B_est,lambda = lasso_p_B_train$lambda.min, family='binomial')
pscore_A <- predict(fit_p_B_est, covariates_obs_A, type = 'response')
pscore_A_train <- predict(fit_p_B_est, covariates_obs_A_train, type = 'response')
pscore_A_est <- predict(fit_p_B_est, covariates_obs_A_est, type = 'response')


########################  Average Treatment Effects (ATE)  ########################

# Generate Modified Outcome
Y_star_A = invisible(y1hat_A - y0hat_A + treat_obs_A*(earnings_obs_A - y1hat_A)/pscore_A 
            - (1-treat_obs_A)*(earnings_obs_A - y0hat_A)/(1-pscore_A))

Y_star_B = invisible(y1hat_B - y0hat_B + treat_obs_B*(earnings_obs_B - y1hat_B)/pscore_B 
            - (1-treat_obs_B)*(earnings_obs_B - y0hat_B)/(1-pscore_B))

Y_star = invisible(0.5*(mean(Y_star_A) + mean(Y_star_B)))

# Average Treatment Effect (ATE)
ATE <- round(mean(Y_star), digits=3)
N = length(Y_star_A) + length(Y_star_B)
SD_ATE <- round(sqrt(0.5*(var(Y_star_A) + (mean(Y_star_A) - mean(Y_star))^2 
                     + var(Y_star_B) + (mean(Y_star_B) - mean(Y_star))^2)/N),digits=3)
print(paste0("Average Treatment Effect (ATE): ", ATE))
print(paste0("Standard Error for ATE: ", SD_ATE))


# Compare results with OLS on full sample
ols <- lm(formula = EARNY4 ~ assignment, data = df)
summary(ols)

#####################  Average Treatment Effects on Treated (ATET)  #####################

p_A = mean(pscore_A)
p_B = mean(pscore_B)

# Generate Modified Outcome
Y_star_A = ????

Y_star_B = ????

Y_star = invisible(0.5*(mean(Y_star_A) + mean(Y_star_B)))

# Average Treatment Effect for Treated (ATET)
ATET <- round(mean(Y_star), digits=3)
                     
var_A = mean((Y_star_A- treat_obs_A*mean(Y_star_A)/p_A)^2)
var_B = mean((Y_star_B - treat_obs_B*mean(Y_star_B)/p_B)^2)
             
N = length(Y_star_A) + length(Y_star_B)
SD_ATET <- round(sqrt(0.5*(var_A + (mean(Y_star_A) - mean(Y_star))^2 
                     + var_B + (mean(Y_star_B) - mean(Y_star))^2)/N),digits=3)
print(paste0("Average Treatment Effect for Treated (ATET): ", ATET))
print(paste0("Standard Error for ATET: ", SD_ATET))

############################################################################


