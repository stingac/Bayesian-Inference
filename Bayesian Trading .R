###############################################################################
# Bayesian inference determined stop-loss in a Long-Short Equity Trading Strategy 
###############################################################################
# Back-test time frame: Backtested using data from 2014 to 2020 due to missing data issues 
# Data: 50 random stocks from S&P500 used in simulation for computational efficiency 
# Trading Strategy: Long on Long top 5 stocks with higher returns (equal weightings)
# and Short bottom 5 stocks with lowest returns during previous month 
# Stop-loss: if posterior_mean (of stock price) is lower than linear prediction
# of stock price then exit position of stock 
# Linear Predictions: Bootstrapped mean returns of previous months as forecast for next month 
# with equal incremental gains between the first day of month to end of month 
# Bayesian Inference: Priors using the predicted stock price in a month and posteriors updated 
# weekly using price data between beginning of month and current week of prior. 

#*****************************************************************
# Libraries and software 
#******************************************************************  
# RStan Installation: Run Once for installation 
remove.packages("rstan")
if (file.exists(".RData")) file.remove(".RData")
install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Rethinking Installation: Run Once for installation 
install.packages(c('devtools','coda','mvtnorm'))
library(devtools)
install_github("rmcelreath/rethinking")

# Listed Libraries
library(rstan)
library(rethinking)
library(ggplot2) 
library(quantmod)
library(PerformanceAnalytics)
library(xts)
library(rvest) #web scrape
library(dplyr)

#*****************************************************************
# Import historical data
#******************************************************************  
# Individual stock data 
setwd("/Users/samuel/Google Drive/Bayesian Inference Stop Loss")
raw_data <-  read.csv('stock_prices.csv',header=TRUE)
raw_historical_data <- read.csv('stock_historical_monthly_returns.csv',header=TRUE)

getSymbols("ADBE;AMZN;BMY;CHTR;FB;FISV;NFLX;NVDA;QCOM;XOM", from = "2013-00-00", to = "2020-01-01")
equity_data <- na.omit(merge(ADBE,AMZN,BMY,CHTR,FB,FISV,NFLX,NVDA,QCOM,XOM))
equity_data_close <- subset(equity_data, select = c(ADBE.Close,AMZN.Close,BMY.Close,CHTR.Close,FB.Close,FISV.Close,NFLX.Close,NVDA.Close,QCOM.Close,XOM.Close))
# equity_data_close <- as.data.frame(equity_data_close)

charts.PerformanceSummary(dailyReturn(ADBE))

# S&P500 stock data 
set.seed(102)
sp_500 <- read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies") %>%
  html_node("table.wikitable") %>%
  html_table() %>%
  select(`Symbol`, Security, `GICS Sector`, `GICS Sub Industry`) %>%
  as_tibble()
sp_500_symbols <- sp_500$Symbol
sp_500_random50 <- c()
sp_500_random50$ticker <- sample(sp_500_symbols,size = 50, replace = FALSE)

#*****************************************************************
# Analysis of priors 
#******************************************************************  
# Use of historical data to see volatility of each stock 
raw_historical_data_no_date <- raw_historical_data[,-1]
std_deviations <- c() 
stock <- c()
for(i in 1:dim(raw_historical_data_no_date)[2]) {
  sd_temp <- round(sd(raw_historical_data_no_date[,i]),4)
  stock <- c(stock,colnames(raw_historical_data_no_date)[i])
  std_deviations <- c(std_deviations, sd_temp)
}
std_dev_table <- as.data.frame(std_deviations)
std_dev_table <- t(std_dev_table)
colnames(std_dev_table) <- stock 
View(std_dev_table)

# Low volatility: FISV, BMY, QCOM, XOM 
volatility_mean_1 <- 1.1
volatility_std_1 <- 0.05 
# Medium volatility: ADBE, NVDA, NFLX, CHTR, FB
volatility_mean_2 <- 1.2 
volatility_std_2 <- 0.1
# High volatility: AMZN
volatility_mean_3 <- 1.3
volatility_std_3 <- 0.2 

# Classifying each stock based on their price volatilities 
classification <- c("L","H","L","M","M","L","M","M","L","L")
std_dev_table <- rbind(std_dev_table, classification)

# Function to set Prior Mean and Prior Standard Deviation for any given stock 
class_prior_set <- function(classification,stock_index) {
  if (classification == "L") { 
    prior_mean <- backtesting_data[1,stock_index]*volatility_mean_1
    prior_mean_std <- prior_mean*0.05
    
    prior_sigma <- backtesting_data[1,stock_index]*volatility_std_1
    prior_sigma_std <- prior_sigma*0.05
  }
  else if (classification == "M") { 
    prior_mean <- backtesting_data[1,stock_index]*volatility_mean_2
    prior_mean_std <- prior_mean*0.05
    
    prior_sigma <- backtesting_data[1,stock_index]*volatility_std_2
    prior_sigma_std <- prior_sigma*0.05
  }
  else if (classification == "H") {
    prior_mean <- backtesting_data[1,stock_index]*volatility_mean_3
    prior_mean_std <- prior_mean*0.05
    
    prior_sigma <- backtesting_data[1,stock_index]*volatility_std_3
    prior_sigma_std <- prior_sigma*0.05
  }
  else {
    print("No classification specified.")
  }
  list <- c(prior_mean,prior_mean_std,prior_sigma,prior_sigma_std)
  return(list)
}

#*****************************************************************
# Analysis of Posteriors 
#******************************************************************  
# Test for one stock Netflix  # Works 
prior_mean <- raw_data$NFLX[1]*1.2
prior_mean_std <- prior_mean*0.05
prior_sigma <- raw_data$NFLX[1]*0.1
prior_sigma_std <- prior_sigma*0.05

s <- sprintf("NFLX ~ dnorm(mu, sigma);
    mu <- alpha;
    alpha ~ dnorm(%f, %f);
    sigma ~ dnorm(%f, %f)", prior_mean, prior_mean_std, prior_sigma, prior_sigma_std)
temp_list <- lapply(strsplit(s, ";")[[1]], function(x) parse(text = x)[[1]])
print(temp_list)

y = c(backtesting_data$NFLX[2])
m1 <- map2stan(temp_list,
  data=list(y=y), start=list(alpha=prior_mean,sigma=prior_sigma), 
  iter=4000 , warmup=1000, chains = 2, cores = 1)
precis(m1)

post_alpha_mean <- mean(extract.samples(m1)$alpha)
post_alpha_std <- sd(extract.samples(m1)$alpha)
post_sigma_mean <- mean(extract.samples(m1)$sigma)
post_sigma_std <- sd(extract.samples(m1)$sigma)

post_prediction <- post_alpha_mean 

### Run two 
y = c(backtesting_data$NFLX[5], backtesting_data$NFLX[6], backtesting_data$NFLX[7],
      backtesting_data$NFLX[8], backtesting_data$NFLX[9],backtesting_data$NFLX[10])
m2 <- map2stan(temp_list,
               data=list(y=y), start=list(alpha=prior_mean,sigma=prior_sigma), 
               iter=4000 , warmup=1000, chains = 2, cores = 1)
precis(m2)
coeftab(m1)

#### Diagnostics #### 
precis(m8.3)
pairs(m8.3)
plot(m8.3)

#*****************************************************************
# Prior/Posterior Visualisations 
#******************************************************************  
par(mfrow = c(1,1))

nflx_data <-data.frame(extract.samples(m1)$NFLX)
colnames(nflx_data) <- "prices"
### Plots ### 
ggplot(nflx_data, aes(x = prices)) + 
  geom_histogram(binwidth=11, colour = "black", fill = "light blue") + 
  labs( x = "Frequency", y = "Frequency") + 
  ggtitle("Prices: Posterior vs Prior") + 
  stat_function(fun= dnorm, args = list(mean=prior_mean,sd=prior_sigma),
                size = 1.5, colour = "light green")+ 
  theme(panel.grid.major.x = element_blank(),panel.grid.major.y = element_line( size=.1, color="black"), axis.line = element_line(colour = "black"),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),panel.border = element_blank(),panel.background = element_blank()) 

# Posterior vs Prior Plot 
x_lower_norm <- 200
x_upper_norm <- 600
ggplot(data.frame(x = c(x_lower_norm , x_upper_norm)), aes(x = x)) + 
  xlim(c(x_lower_norm , x_upper_norm)) + 
  labs( x = "x", y = "f(x)") + 
  ggtitle("Density: Posterior vs Prior") +  
  stat_function(fun = dnorm, args = list(mean = prior_mean, sd = prior_sigma), size = 1, aes(colour = "mean")) + 
  stat_function(fun = dnorm, args = list(mean = post_alpha_mean, sd = post_sigma_mean), size = 1, aes(colour = "mean")) + 
  scale_color_manual("Mean & Std. Deviation \n Parameters", values = c("blue", "red", "red")) + 
  theme(panel.grid.major.x = element_blank(),panel.grid.major.y = element_line( size=.1, color="black"), axis.line = element_line(colour = "black"),panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),panel.border = element_blank(),panel.background = element_blank()) 

#*****************************************************************
# Positions (Rankings of Stocks)
#******************************************************************  

# Variables and progression bar
n <- length(sp_500_random50$ticker)
pb <- txtProgressBar(min = 0, max = n, style=3)
random50_data <- xts() 

# Extract adjusted price data for the 50 random S&P500 stocks 
# Source code: Joshua Ulrich @stakexchange.com
for(i in 1:n) {
  symbol <- sp_500_random50$ticker[i]
  # specify the "from" date to desired start date
  tryit <- try(getSymbols(symbol,from="2013-01-01",start ="2020-01-01", src='yahoo'))
  if(inherits(tryit, "try-error")){
    i <- i+1
  } else {
    # specify the "from" date to desired start date
    data <- getSymbols(symbol, from="2013-01-01", start ="2020-01-01", src='yahoo')
    random50_data <- merge(random50_data, Ad(get(sp_500_random50$ticker[i])))
    rm(symbol)
  }
  setTxtProgressBar(pb, i)
}

# Create tables 
# Price Data 
random50_data <- random50_data[index(random50_data) > '2013-12-31']

# Daily Returns 
random50.dailyreturns <- ROC(random50_data)
random50.dailyreturns.table <- c()

# Monthly Returns 
random50.monthlyreturns <- lapply(1:n, function(x) {monthlyReturn(random50_data[,x])})
random50.monthlyreturns.table <- c()
for(i in 1:n) {random50.monthlyreturns.table <- cbind(random50.monthlyreturns.table,random50.monthlyreturns[[i]]) }
random50.monthlyreturns.table <- random50.monthlyreturns.table[index(random50.monthlyreturns.table) > '2013-12-31']

# Weekly Returns 
random50.weeklyreturns <- lapply(1:n, function(x) {weeklyReturn(random50_data[,x])})
random50.weeklyreturns.table <- c()
for(i in 1:n) {random50.weeklyreturns.table <- cbind(random50.weeklyreturns.table,random50.weeklyreturns[[i]]) }
random50.weeklyreturns.table <- random50.weeklyreturns.table[index(random50.weeklyreturns.table) > '2013-12-31']

# Quantiles of returns 
qqDown = quantile(unlist(random50.monthlyreturns.table[1,]),prob = 0.1)
qqUp =  quantile(unlist(random50.monthlyreturns.table[1,]),prob = 0.9)

# Creating Monthly Weight/ Position table Long/Short  = [1,-1]
random50.monthlyWeights <- c()

for(r in 1:dim(random50.monthlyreturns.table)[1]) {
  
  temp_row_weights <- c() 
  
  temp_qqDown = quantile(unlist(random50.monthlyreturns.table[r,]),prob = 0.1)
  temp_qqUp =  quantile(unlist(random50.monthlyreturns.table[r,]),prob = 0.9)
  
  for(c in 1:dim(random50.monthlyreturns.table)[2]) { 
    if(random50.monthlyreturns.table[r,c] < temp_qqDown) {
      temp_weight = - 1 
    }
    else if (random50.monthlyreturns.table[r,c] > temp_qqUp) {
      temp_weight = 1 
    }
    else { temp_weight = 0}
    temp_row_weights <- c(temp_row_weights, temp_weight)
  }
  random50.monthlyWeights <- rbind(random50.monthlyWeights, temp_row_weights)
}

random50.monthlyWeights <- xts(random50.monthlyWeights, order.by = index(random50.monthlyreturns.table))

# Creating Daily Weight/ Position table Long/Short  = [1,-1]
random50.dailyWeights <- random50.dailyWeights[index(random50.dailyWeights) > '2013-12-31']

# Transfer monthly weights to daily weights 
date_list <- format(index(random50.monthlyWeights),"%Y-%m")
# n = number of randomly picked stocks 
for(year_month in date_list) { 
  for(c in 1:n) {
    random50.dailyWeights[year_month,c] <- random50.monthlyWeights[year_month,c]
  }
}

#*****************************************************************
# Linear Predictions 
#******************************************************************  
random50.predictions <- random50.dailyreturns

# Bootstrap simulation function 
simulation <- function(value1, simulations, temp.Stock.week){
  forecast_value_sum <- 0 
  for(i in 1:simulations) {
    return1.samp <- sample(as.vector(temp.Stock.week), 4, replace = TRUE)
    return2.samp <- 1 + return1.samp
    return3.samp <- cumprod(return2.samp)
    value2 <- return3.samp*as.numeric(value1)
    forecast_value_sum <- forecast_value_sum + last(value2)
  }
  return(forecast_value_sum)
}

# Calculation of incremental day gains for particular month 
incremental_day_gain <- function(forecast_value_sum, year_month, value1, simulations) { 
  month_forecast_value <- 0 
  month_forecast_value <- forecast_value_sum/simulations 
  temp_days <- count(index(random50.predictions[year_month]))
  incr_day_gain <- (month_forecast_value - value1)/temp_days 
  return(incr_day_gain)
}

# Predictions calculation 
monthly_date_list <- index(random50.monthlyreturns.table)
simulations <- 500

for(c in 1:n) {
  temp.Stock.week <- random50.weeklyreturns.table[,c]
  
  for(year_month_index in 1:len(monthly_date_list)) {
    year_month <- format(monthly_date_list[year_month_index],"%Y-%m")
    value1 <- first(random50_data[year_month,c])

    forecast_value_sum  <- simulation(value1, simulations, temp.Stock.week)

    incr_day_gain <- incremental_day_gain(forecast_value_sum, year_month, value1, simulations)
    
    daily_date_list <- index(random50.dailyreturns[year_month])
    for(year_month_day_index in 1:len(daily_date_list)){
      year_month_day <- format(daily_date_list[year_month_day_index],"%Y-%m-%d")
      if(year_month_day == first(index(random50_data[year_month]))) { 
        random50.predictions[year_month_day,c] <- value1
      }
      else { 
        day_index <- which(year_month_day == index(random50_data[year_month]))
        random50.predictions[year_month_day,c] <- value1+incr_day_gain*day_index
      }
    }
  }
}

# Plotting for report 
temp.Stock.week <- weeklyReturn(equity_data_close$ADBE.Close)
plot(temp.Stock.week)
# Create a plot window for results
value1 <- last(equity_data_close['2019-12',1])
plot(NULL,
     xlim = c(2019.75,2020),
     ylim = c(200,500),
     xlab = "Time",
     ylab = "Value")
abline(h = value1, col = "blue", lty = 2)
# abline(h = 150, col = "red", lty = 2 ) #your buy price

forecast_value <- 0 
forecast_value_sum <- 0
simulations <- 1000
for(i in 1:simulations) {
  return1.samp <- sample(as.vector(temp.Stock.week), 12, replace = TRUE)
  return2.samp <- 1 + return1.samp
  return3.samp <- cumprod(return2.samp)
  value2 <- return3.samp*as.numeric(value1)
  value3 <- ts(value2, start = c(2019,42),frequency = 52)
  lines(value3, col = "grey")
  forecast_value_sum <- forecast_value_sum + last(value3)
}
forecast_value <- forecast_value_sum/simulations

#*****************************************************************
# Prior/Posterior predictions 
#******************************************************************  
weekly_date_list <- index(random50.weeklyreturns.table)
random50.posterior <- random50.weeklyreturns.table
random50.posterior[] <- 0 

# Posterior function: returns posterior_mean
posterior_mean <- function(update_data, prior_mean, prior_mean_std, prior_sigma, prior_sigma_std) {
  s <- sprintf("stock_price ~ dnorm(mu, sigma);
    mu <- alpha;
    alpha ~ dnorm(%f, %f);
    sigma ~ dnorm(%f, %f)", prior_mean, prior_mean_std, prior_sigma, prior_sigma_std)
  temp_list <- lapply(strsplit(s, ";")[[1]], function(x) parse(text = x)[[1]])
  
  y = coredata(update_data)
  m1 <- map2stan(temp_list,
                 data=list(y=y), start=list(alpha=prior_mean,sigma=prior_sigma), 
                 chains=2, cores=1 , iter=2000 , warmup=500)
  
  post_alpha_mean <- mean(extract.samples(m1)$alpha)
  post_prediction <- post_alpha_mean 
  return(post_prediction)
}

# Posterior Predictions creation
pb <- txtProgressBar(min = 0, max = n, style=3)
i <- 0 
for(c in 1) {
  i <- i+1
  for(weekly_index in 1:len(weekly_date_list)){
    year_month <- format(weekly_date_list[weekly_index],"%Y-%m")
    if(first(random50.dailyWeights[year_month,c]) != 0) {
       
      # defining priors for normal distribution 
      prior_mean <- last(random50.predictions[year_month,c])
      prior_mean_std <- prior_mean*0.3
      prior_sigma <- last(random50.predictions[year_month,c])*0.2
      prior_sigma_std <- prior_sigma*0.3
      
      # price data for update includes all prices before weekly_index date within month 
      update_data <- random50_data[index(random50_data) <= weekly_date_list[weekly_index] & index(random50_data) >= first(index(random50_data[year_month])), c]
    
      # updating priors for posterior prediction and returning posterior mean 
      temp_post_prediction <- posterior_mean(update_data, prior_mean, prior_mean_std, prior_sigma, prior_sigma_std)
      
      # updating posterior predictions table 
      random50.posterior[weekly_date_list[weekly_index],c] <- temp_post_prediction
    }
  }
  setTxtProgressBar(pb, i)
}


#*****************************************************************
# Stop-loss function 
#******************************************************************  
# 1 returns an exit 
# Posterior mean <  prediction at time t then exit long position  
# Posterior mean > prediction at time t then exit short position 

posterior.stop <- function(weight,price,tstart,tend,pstop, predictions, posterior.means) {
  index = tstart : tend
  if(weight > 0)
    if(prediction[index] > posterior.means[index])
      return(1)
  else if (weight < 0)
    if(prediction[index] < posterior.means[index])
      return(1)
  else if (weight == 0)
    return(0)
}

#*****************************************************************
# Back-test 
#******************************************************************  
models = list()

# Analysis: first Stock of random50 
data <- new.env()
data$prices <- random50_data[,1]

# ensure data replicates Working code data env 

date$weight[] = NA
data$weight[] = custom.stop.fn(coredata(random50.dailyWeights[,1]), coredata(random50_data[,1]), posterior.stop, 
                               random50.predictions[,1] ,random50.posterior[,1])

# bt.run.share requires an environment with prices and weights
models$one.stock = bt.run.share(data, clean.signal=T, trade.summary = TRUE)

# portfolio Stocks: store each stock in models() 

#*****************************************************************
# Risk Metrics 
#******************************************************************  
# 1. Return 
# 2. Sharpe Ratio
# 3. Max Drawdown 
# 4. Max Drawdown Duration 
# 5. Lowest Value from Investment 
# 6. Time in Market Ratio 

# Performance(myReturns)
Performance <- function(x) {
  cumRetx = Return.cumulative(x)
  annRetx = Return.annualized(x, scale=252)
  sharpex = SharpeRatio.annualized(x, scale=252)
  winpctx = length(x[x > 0])/length(x[x != 0])
  annSDx = sd.annualized(x, scale=252)
  
  DDs <- findDrawdowns(x)
  maxDDx = min(DDs$return)
  maxLx = max(DDs$length)
  
  Perf = c(cumRetx, annRetx, sharpex, winpctx, annSDx, maxDDx, maxLx)
  names(Perf) = c("Cumulative Return", "Annual Return","Annualized Sharpe Ratio",
                  "Win %", "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown")
  return(Perf)
}


