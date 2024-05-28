library(RSQLite)
library(tidyverse)
library(scales)
library(furrr)
library(broom)
library(tidymodels)
library(glmnet)
library(timetk)
library(keras)
library(ggplot2)
library(ranger)
library(hardhat)
library(lmtest)
library(sandwich)
library(plotly)
library(patchwork)
library(gtools)
library(future.apply)
library(philentropy)
library(ggridges)
library(frenchdata)

                                                                        ###### DISCLAIMER #######
                          ### This script relies on the output from the Machine learning model predictions, Average daily dollar volume and Risk free rate scripts 
                          ### It produces all the backtesting results of our paper. The scripts uses the data file tidy_finance_ML.sqlite that can be obtained with 
                          ### acces to CRSP and by following https://voigtstefan.quarto.pub/course-exercises/gu_kelly_xiu.html.
                          ### Set working directory to where the output from previous scripts are store alongside tidy_finance_ML.sqlite
                          ### Part of the code is run in parallel using all available cores


####################################### Loading and preparing data  ##########################################


### Load data set from ML predictions ###

# Make a list of the relevant csv files
csv_files <- list.files(pattern = "MLdata_\\d+\\.csv")
csv_files <- mixedsort(csv_files)

# Read the first CSV file (2005 prediction) to initialize the data frame
collective_predicted_values <- read.csv(csv_files[1])

# Loop through the remaining CSV files and append them to the data frame
for (file in csv_files[-1]) {
  data <- read.csv(file)
  collective_predicted_values <- rbind(collective_predicted_values, data)
}


### Load data set for risk free rate (rf) ###

# Choose csv file containing rf
rf_data <- read.csv("rfdata.csv") |>
  select(permno, month, rf)


### Load data set for average dollar volume (dolvol) ###

# Choose csv file containing dolvol
dolvol_data <- read.csv("CRSPdolvol.csv") |>
  select(permno, month, avg_dolvol)

#Force R to read month as date
collective_predicted_values$month <- as.Date(collective_predicted_values$month)
rf_data$month <- as.Date(rf_data$month)


### Combining datasets ###

# Merge data sets
collective_predicted_values <- merge(collective_predicted_values, rf_data, by = c("month","permno")) 
collective_predicted_values <- merge(collective_predicted_values, dolvol_data, by = c("month","permno")) 

# Remove firms where the average dollar volume is 0
collective_predicted_values <- collective_predicted_values[collective_predicted_values$avg_dolvol != 0, ]

# In order to do calculations of changes in weights across months, we need to make sure that the data set has the same size for each month
# Thus, we add rows such that all permnos are represented in each month, even though these are not listed in that specific month

# Create a reference dataset with all permnos across all months
all_permnos <- data.frame(permno = unique(collective_predicted_values$permno))

# Create a combination of all months and permnos
all_combinations <- expand.grid(month = unique(collective_predicted_values$month), permno = unique(all_permnos$permno))

# Merge to include all permnos in each month
merged_data <- merge(all_combinations, collective_predicted_values, by = c("month", "permno"), all.x = TRUE)

# All the added permnos in each month will not have values for ret_excess and avg_dolvol
# To allow for calculation later, we set ret_excess to zero for NA's and dolvol to inf. (hereby assuming that exiting companies that get delisted is free/No TC) 
merged_data$ret_excess[is.na(merged_data$ret_excess)] <- 0
merged_data$avg_dolvol[is.na(merged_data$avg_dolvol)] <-1e10000000


# Calculate market return
market_return_df <- collective_predicted_values |>
  group_by(month) |>
  summarize(market_ret = weighted.mean(ret_excess, mktcap_lag))

# Add market return to dataset
merged_data <- left_join(merged_data, market_return_df, by = "month") 
  
write.csv(market_return_df, "market_return.csv", row.names = FALSE)



######################################### Evaluating predictive performance of ML model ###################################################


# Calculate R-squarred for all firms
R_squarred_data <- collective_predicted_values |>
  select(pred, ret_excess, month, mktcap_lag) |>
  mutate(SS_tot = ret_excess^2,
         SS_res = (ret_excess-pred)^2 
  )

R_squarred = 100*( 1 - sum(R_squarred_data$SS_res)/sum(R_squarred_data$SS_tot))

#Print
R_squarred


# Sort data into mkt cap decile
collective_predicted_values <- collective_predicted_values |>
  group_by(month) |>
  mutate(decile = ntile(mktcap_lag, 10))

# Calculate R-squared for each decile
R_squared_data <- collective_predicted_values |>
  group_by(decile) |>
  summarise(
    SS_tot = sum(ret_excess^2),
    SS_res = sum((ret_excess - pred)^2),
    R_squared = 100 * (1 - SS_res / SS_tot)
  ) |>
  ungroup()

# Print
print(R_squared_data)



############################################ CREATING FUNCTIONS FOR PORTFOLIO CHOICE AND EVALUATION ####################################################

### Creating function for choosing monthly weights ###    
portfolio_weights <- function(data,
                              current_month,
                              wealth,             # Sets portfolio size
                              Old_weights_data,
                              equal_weighting,    # FALSE for value weighting and true for equal weighting
                              portfolio_type,     # Sets long or short portfolio
                              Agnostic_of_TC,     # If TRUE: sorts based on expected gross returns. If FALSE: Sorts based on expected net returns
                              Sort_type) {        # Top/buttom 50 for choosing 50 firms in each of long and short. Top/buttom 10% for choosing top/buttom top 10% instead.

  data |>
    # Set month
    filter(month == current_month) |>
    
    #Add old weights from previous month
    left_join(Old_weights_data, by = "permno") |>
    
    mutate(
      
      #Choose weights for TC predictions based on sorting type (only for cost-sensitve)
      weight_TC = case_when(
        Sort_type == "Top/buttom 10%" ~ 1/ceiling(0.1*sum(!is.na(pred))),
        Sort_type == "Top/buttom 50" ~ 1/50

      ),
      
      # Choose prediction to use for sorting (with or without TC/for agnostic or cost-sensitve).
      pred = case_when(
        Agnostic_of_TC == FALSE & portfolio_type == "Long" ~ pred - (8.89/1000 *  (wealth * abs(weight_TC-Old_weights) / avg_dolvol )^(1/2) - 8.89/1000 *  (wealth * (Old_weights) / avg_dolvol )^(1/2)),
        Agnostic_of_TC == FALSE & portfolio_type == "Short" ~ pred + (8.89/1000 *  (wealth * abs(weight_TC-Old_weights) / avg_dolvol )^(1/2) - 8.89/1000 *  (wealth * (Old_weights) / avg_dolvol )^(1/2)),
        Agnostic_of_TC == TRUE ~ pred
      ),
      
      # Choose portfolio long/short portfolio based on Sort_type
      portfolio = case_when(
        Sort_type == "Top/buttom 10%" &
          ((portfolio_type == "Long" & pred >= quantile(pred, 0.9, na.rm = TRUE)) |
             (portfolio_type == "Short" & pred <= quantile(pred, 0.1, na.rm = TRUE))) ~ portfolio_type,
        Sort_type == "Top/buttom 50" &
          ((portfolio_type == "Long" & rank(desc(pred)) <= 50) |
             (portfolio_type == "Short" & rank(pred) <= 50)) ~ portfolio_type,
        TRUE ~ "other"
      ),
      

      # Set weights (equal or value weighting)
      weight = case_when(
        equal_weighting == TRUE ~ if_else(portfolio == portfolio_type, 1/sum(portfolio == portfolio_type), 0),
        equal_weighting == FALSE ~ if_else(portfolio == portfolio_type, mktcap_lag/sum(mktcap_lag[portfolio == portfolio_type]), 0)
      ),
      
      
      # Calculate TC pr stock in percent
      TC = 8.89/1000 *  (wealth * abs(weight-Old_weights) / avg_dolvol )^(1/2),
      
      # Calculate TC contribution from each stock to portfolio TC in pct points
      Weighted_TC = TC * abs(weight-Old_weights),
      
      # Calculate turnover pr stock
      turnover = abs(weight-Old_weights),
      
      # Calculate next periods initial weights
      Old_weights = ifelse(is.na(rf), 0, weight * (1 + rf + ret_excess - TC) / (1 + market_ret + rf))
      
    )
  
}



### Creating function for storing monthly performance of chosen portfolio ### 
store_monthly_portfolio <- function(data, 
                               portfolio_type, # Sets long or short portfolio
                               scenario,       
                               Agnostic_of_TC,
                               current_month,
                               gross) # TRUE to calculate gross returns. FALSE for net returns
  {
  
  
  # Set label of agnostic or cost-sensitve
  agnostic_label <- case_when(Agnostic_of_TC == TRUE ~ "Agnostic",
                              Agnostic_of_TC == FALSE ~ "Cost-sensitve")
  
  data |>
    summarise(
      
      net_return = if (gross==FALSE) {
        # Calculate net return (with transaction costs)
        case_when(
          portfolio_type == "Long" ~ weighted.mean(ret_excess, weight) - sum(Weighted_TC),
          portfolio_type == "Short" ~ weighted.mean(ret_excess, weight) + sum(Weighted_TC)
        )
      } else {
        # Calculate gross return (without transaction cost)
        weighted.mean(ret_excess, weight)
      },
      
      # Calculate turnover
      turnover = sum(turnover),
      
      # Calculate avg mktcap of portfolio relative to mean mktcap of all firms
      avg_mktcap_share = mean(mktcap_lag[portfolio == portfolio_type]/mean(mktcap_lag[!is.na(mktcap_lag)])),
      
      #Store month
      month = as.Date(current_month),
      
      #Store portfolio type
      portfolio = portfolio_type,
      
      # Store scenario
      scenario = scenario,
      
      # Store if agnostic or cost-senstive
      agnostic_of_TC = agnostic_label)
  
}


################# SHARPE ERRORS - The following code is taken from Ledoit andWolf (2008  ###############

hac <- function(ret, digits = 3)
{
  ret1 = ret[, 1]
  ret2 = ret[, 2]
  mu1.hat = mean(ret1)
  mu2.hat = mean(ret2)
  sig1.hat = sd(ret1)
  sig2.hat = sd(ret2)
  SR1.hat = mu1.hat / sig1.hat
  SR2.hat = mu2.hat / sig2.hat
  SRs = round(c(SR1.hat, SR2.hat), digits)
  diff = SR1.hat - SR2.hat
  names(SRs) = c("SR1.hat", "SR2.hat")
  se = compute.se.Parzen(ret)
  PV = 2 * pnorm(-abs(diff)/se)
  list(Sharpe.Ratios = SRs, Difference = round(diff, digits),
       Standard.Errors = se, p.Value = PV)
}

compute.se.Parzen <- function(ret)
{
  ret1 = ret[, 1]
  ret2 = ret[, 2]
  T = length(ret1)
  mu1.hat = mean(ret1)
  mu2.hat = mean(ret2)
  ret1.2 = ret1^2
  ret2.2 = ret2^2
  gamma1.hat = mean(ret1.2)
  gamma2.hat = mean(ret2.2)
  gradient = rep(0, 4)
  gradient[1] = gamma1.hat/(gamma1.hat - mu1.hat^2)^1.5
  gradient[2] = -gamma2.hat/(gamma2.hat - mu2.hat^2)^1.5
  gradient[3] = -0.5 * mu1.hat/(gamma1.hat - mu1.hat^2)^1.5
  gradient[4] = 0.5 * mu2.hat/(gamma2.hat - mu2.hat^2)^1.5
  V.hat = cbind(ret1 - mu1.hat, ret2 - mu2.hat, ret1.2 - gamma1.hat, ret2.2 - gamma2.hat)
  Psi.hat = compute.Psi.hat(V.hat)
  se = as.numeric(sqrt(t(gradient) %*% Psi.hat %*% gradient/T))
  se
}


compute.Psi.hat <- function(V.hat) 
{
  T = length(V.hat[, 1])
  alpha.hat = compute.alpha.hat(V.hat)
  S.star = 2.6614 * (alpha.hat * T)^0.2
  S.star = min(S.star, T-1)
  Psi.hat = compute.Gamma.hat(V.hat, 0)
  j = 1
  while (j < S.star) {
    Gamma.hat = compute.Gamma.hat(V.hat, j)
    Psi.hat = Psi.hat + kernel.Parzen(j/S.star) * (Gamma.hat + t(Gamma.hat))
    j = j + 1
  }
  Psi.hat = (T/(T - 4)) * Psi.hat
  Psi.hat
}

compute.Gamma.hat <- function (V.hat, j) 
{
  dimensions = dim(V.hat)
  T = dimensions[1]
  p = dimensions[2]
  Gamma.hat = matrix(0, p, p)
  if (j >= T) 
    stop("j must be smaller than the row dimension!")
  for (i in ((j + 1):T)) Gamma.hat = Gamma.hat + V.hat[i, ] %*% 
    t(V.hat[i - j, ])
  Gamma.hat = Gamma.hat/T
  Gamma.hat
}

compute.alpha.hat <- function(V.hat) 
{
  dimensions = dim(V.hat)
  T = dimensions[1]
  p = dimensions[2]
  numerator = 0
  denominator = 0
  for (i in (1:p)) {
    fit = ar(V.hat[, i], 0, 1, method = "ols")
    rho.hat = as.numeric(fit[2])
    sig.hat = sqrt(as.numeric(fit[3]))
    numerator = numerator + 4 * rho.hat^2 * sig.hat^4/(1 - rho.hat)^8
    denominator = denominator + sig.hat^4/(1 - rho.hat)^4
  }
  numerator/denominator
}


kernel.Parzen <- function(x) 
{
  if (abs(x) <= 0.5) 
    result = 1 - 6 * x^2 + 6 * abs(x)^3
  else if (abs(x) <= 1) 
    result = 2 * (1 - abs(x))^3
  else
    result = 0
  result
}

####################################################################################################

### Creating function for evaluating portfolios ###               
portfolio_evaluation <- function(data, portfolio, scenario, Agnostic_label) 
{
  
  
  if (Agnostic_label == "NO") {
    market_data <- data |>
      filter(portfolio == "Market portfolio", scenario == !!scenario)
    
    filtered_data <- data |>
      filter(portfolio == !!portfolio, scenario == !!scenario)
  } else {
    market_data <- data |>
      filter(portfolio == "Market portfolio", scenario == !!scenario, agnostic_of_TC == !!Agnostic_label)
    
    filtered_data <- data |>
      filter(portfolio == !!portfolio, scenario == !!scenario, agnostic_of_TC == !!Agnostic_label)
  }
  
  filtered_data$market_ret <- market_data$net_return
  
  CAPM_M_fit <- lm(net_return ~ market_ret, data = filtered_data)
  coef_test <- coeftest(CAPM_M_fit, vcov = NeweyWest)
  
  SRtest_data <- cbind(filtered_data$net_return, filtered_data$market_ret)
  
  hac_comp <- hac(SRtest_data)
  
  mkt_sharp_ann <- hac_comp$Sharpe.Ratios[2]*sqrt(12)
  
  SR_dff_ann <- hac_comp$Difference*sqrt(12)
  
  SR_ub <- SR_dff_ann + qnorm(0.975)*hac_comp$Standard.Errors*sqrt(12) + mkt_sharp_ann
  
  SR_lb <- SR_dff_ann - qnorm(0.975)*hac_comp$Standard.Errors*sqrt(12) + mkt_sharp_ann
  
  Output <- filtered_data |>
    summarise(
      
      mean_return = mean(net_return),
      sd_return = sd(net_return),
      annualized_sharpe_ratio = sqrt(12) * mean_return / sd_return,
      annualized_return = mean_return * 12 * 100,
      turnover = mean(turnover),
      avg_mktcap_share = mean(avg_mktcap_share),
      alpha = round(coef_test[1, 1] * 12 * 100, digits = 2),
      portfolio = !!portfolio,
      scenario = !!scenario,
      Agnostic_label =!!Agnostic_label,
      alpha_se = coef_test[1, 2] * sqrt(12) * 100,
      alpha_lb = alpha  - qnorm(0.975)*alpha_se,
      alpha_ub = alpha  + qnorm(0.975)*alpha_se,
      SR_dff_ann =SR_dff_ann,
      SR_ub = SR_ub,
      SR_lb = SR_lb
    )

}

######################################### Computations for main results table and figure ########################################


# Creating different scenarios/specifications
scenarios <- list(
  # Spec for figure 1.1
  scenario_1 = list(wealth_level = 100000000, Agnostic_of_TC = TRUE, equal_weighting = FALSE, Sort_type = "Top/buttom 50", gross = TRUE),
  # Spec for figure 1.2
  scenario_2 = list(wealth_level = 100000000, Agnostic_of_TC = TRUE, equal_weighting = FALSE, Sort_type = "Top/buttom 50", gross = FALSE),
  # Spec for figure 1.3
  scenario_3 = list(wealth_level = 100000000, Agnostic_of_TC = TRUE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = TRUE),
  #Spec for figure 1.4
  scenario_4 = list(wealth_level = 100000000, Agnostic_of_TC = TRUE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = FALSE),
  #Spec for figure 2
  scenario_5 = list(wealth_level = 100000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = TRUE),
  #Spec for figure 2
  scenario_6 = list(wealth_level = 100000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = FALSE)
  
)


# Portfolio backtesting: Looping over the different scenarios and portfolio types

all_results <- data.frame()

for (portfolio_type in c("Long", "Short")) {
  for (scenario_name in names(scenarios)) {
    
    #Set scenario name
    scenario <- scenarios[[scenario_name]]
    
    #Set wealth level 
    wealth <- scenario$wealth_level
    
    #Set initial old weights data 
    Old_weights_data <- merged_data |>
      filter(month == "2005-01-01") |>
      select(permno) |>
      mutate(Old_weights = 0)
    

    # Loop over all the months
    for (current_month in seq(as.Date("2005-01-01"), as.Date("2021-11-01"), by = "months")) {

      
      # Choose weights
      chosen_weights <- portfolio_weights(data = merged_data,
                                          current_month = current_month,
                                          wealth = wealth,
                                          Old_weights_data = Old_weights_data,
                                          equal_weighting = scenario$equal_weighting,
                                          portfolio_type = portfolio_type,
                                          Agnostic_of_TC = scenario$Agnostic_of_TC,
                                          Sort_type = scenario$Sort_type)
      
      
      # Storing monthly performance of chosen portfolio
      results <- store_monthly_portfolio(data = chosen_weights, 
                                    portfolio_type = portfolio_type,
                                    scenario = scenario_name,    
                                    Agnostic_of_TC = scenario$Agnostic_of_TC,
                                    current_month = current_month,
                                    gross = scenario$gross)
      
      # Bind results
      all_results <- bind_rows(all_results, results)
      
      # Store old weights
      Old_weights_data <- chosen_weights |>
        select (permno, Old_weights)
      
      # Collect market return and risk free
      market_ret <- chosen_weights$market_ret[which(!is.na(chosen_weights$market_ret))[1]]
      risk_free <- chosen_weights$rf[which(!is.na(chosen_weights$rf))[1]]
      
      # Update wealth level for next month
      wealth <- wealth * (1 + market_ret + risk_free)
    }
  }
}


# Retrieve market portfolio for each of the scenarios in all_results
market_portfolio_rows <- data.frame(
  month = unique(market_return_df$month),  
  net_return = market_return_df$market_ret[match(unique(market_return_df$month), market_return_df$month)],
  portfolio = "Market portfolio",
  scenario = rep(c("scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5", "scenario_6" ), each = length(unique(market_return_df$month))))


# Add columns in order to have same format as all_results
market_portfolio_rows$turnover <- 0
market_portfolio_rows$avg_mktcap_share <- 1
market_portfolio_rows$agnostic_of_TC <- "Agnostic"


# Append the market portfolio to all_results
merged_all_results <- rbind(all_results, market_portfolio_rows)


# Calculate long short portfolio
long_short_rows <- merged_all_results |>
  group_by(month, scenario, agnostic_of_TC) |>
  summarize(
    net_return = sum(net_return[portfolio == "Long"]) - sum(net_return[portfolio == "Short"]),
    turnover = (turnover[portfolio=="Long"] + turnover[portfolio=="Short"] )/2,
    avg_mktcap_share = (avg_mktcap_share[portfolio=="Long"] + avg_mktcap_share[portfolio=="Short"] )/2
  ) |>
  mutate(portfolio = "Long-Short")   


# Append the Long_short portfolio to all_results
merged_all_results <- bind_rows(merged_all_results, long_short_rows)


# Calculate cumulative log returns for plotting
comulative_returns <- merged_all_results |>
  mutate(net_return_comp = log(net_return + 1)) |>
  group_by(portfolio, scenario) |>
  mutate(net_return_comp = cumsum(net_return_comp)) |>
  ungroup()


# Set names for different scenarions
scenario_names_mapping <- c("scenario_1" = "Agnostic Sort - Value Weighted (Gross Return)",
                            "scenario_2" = "Agnostic Sort - Value Weighted (Net Return)",
                            "scenario_3" = "Agnostic Sort - Equal Weighted (Gross Return)",
                            "scenario_4" = "Agnostic Sort - Equal Weighted (Net Return)",
                            "scenario_5" = "Cost-Sensitive Sort - Equal Weighted (Gross Return)",
                            "scenario_6" = "Cost-Sensitive Sort - Equal Weighted (Net Return)")


# Choose relevant scenarions for figure 1
figure_1_scenarios <- c("scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5", "scenario_6")


# Choose data for figure 1 based on scenario names
figure_1_data <- comulative_returns |>
  filter(scenario %in% figure_1_scenarios)


# Replace scenario names with the above names
figure_1_data$scenario <- scenario_names_mapping[as.character(figure_1_data$scenario)]


# Plotting figure
Plot_1 <-ggplot(data = figure_1_data, aes(x = month, y = net_return_comp, color = portfolio, linetype = portfolio)) +
        geom_line() +
        geom_hline(yintercept = 0, linetype = "solid", color = "gray") + 
        facet_wrap(facets = vars(scenario), scales = "free_y", ncol = 2) +
        scale_color_manual(values = c("Market portfolio" = "black", "Short" = "red", "Long" = "darkgreen", "Long-Short" = "blue")) +
        scale_linetype_manual(values = c("Market portfolio" = "dotted", "Short" = "solid", "Long" = "solid", "Long-Short" = "solid")) +
        ggtitle("") +
        xlab("") +
        ylab("Cumulative Log Return") +
        theme_gray() +
        labs (linetype = "", color = "") +
        theme(plot.title = element_text(hjust = 0.5), axis.line.y = element_line(color = "gray"),
              text = element_text(size = 15), legend.position = "bottom")
Plot_1

  
# Saving the plots locally
ggsave("plot_1.png", plot = Plot_1, bg = "white", dpi = 600, width = 12, height = 10)



# Evaluate each of the portfolios in the different scenarios
main_table <- data.frame()

for (portfolio in c("Long", "Short", "Long-Short", "Market portfolio")) {
  
  for (scenario in c("scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5", "scenario_6")) {
    
    # Call the portfolio_evaluation function 
    evaluation_result <- portfolio_evaluation(merged_all_results, portfolio, scenario, "NO")
    
    # Store the evaluation result
    main_table <- rbind(main_table, evaluation_result)
  }
}

# Print main results table
print(main_table)

#write.csv(main_table, "main_table.csv", row.names = FALSE)

######################################### Make computations for breakeven analysis ##################################################

#Create function to compute portfolios for different wealth levels/scenarios
breakeven <- function(scenario, data) {
  
  all_results <- data.frame()
  
  # Loop over each portfolio type
  for (portfolio_type in c("Long", "Short")) {
    # Loop over each scenario of Agnostic_of_TC
    for (Agnostic_of_TC in c(TRUE, FALSE)) {
      
      
      #Set initial old weights
      Old_weights_data <- data |>
        filter(month == "2005-01-01") |>
        select(permno) |>
        mutate(Old_weights = 0)
      
      
      wealth <-scenario
      
      # Loop over each month
      for (current_month in seq(as.Date("2005-01-01"), as.Date("2021-11-01"), by = "months")) {
        # Choose weights
        chosen_weights <- portfolio_weights(data = data,
                                            current_month = current_month,
                                            wealth = wealth,
                                            Old_weights_data = Old_weights_data,
                                            equal_weighting = TRUE,
                                            portfolio_type = portfolio_type,
                                            Agnostic_of_TC = Agnostic_of_TC,
                                            Sort_type = "Top/buttom 50")
        
        # Storing monthly performance of chosen portfolio
        results <- store_monthly_portfolio(data = chosen_weights, 
                                           portfolio_type = portfolio_type,
                                           scenario = scenario,    
                                           Agnostic_of_TC = Agnostic_of_TC,
                                           current_month = current_month,
                                           gross = FALSE)
        
        
        # Bind results
        all_results <- bind_rows(all_results, results)
        
        # Store old weights
        Old_weights_data <- chosen_weights |>
          select (permno, Old_weights)
        
        # Collect market return and risk free and update wealth
        market_ret <- chosen_weights$market_ret[which(!is.na(chosen_weights$market_ret))[1]]
        risk_free <- chosen_weights$rf[which(!is.na(chosen_weights$rf))[1]]
        
        wealth <- wealth * (1 + market_ret + risk_free)
      }
    }
  }
  
  return(all_results)
  
}


# Initiate parallel computation
plan(multisession)

# Compute portfolios across wealth levels in parallel
all_breakeven_results <- future_lapply(seq(0.00001, 5000000000, by = 10000000), breakeven, data= merged_data)

# Bind results from computations
all_breakeven_results <- do.call(rbind, all_breakeven_results)

# Turn off parallel computation
plan(sequential)


# Retrieve market portfolio for each of the scenarios in all_results
market_portfolio_rows <- data.frame(
  month = rep(unique(market_return_df$month), each = length(seq(0.00001, 5000000000, by = 10000000))),
  net_return = rep(market_return_df$market_ret[match(unique(market_return_df$month), market_return_df$month)], each = length(seq(0.00001, 5000000000, by = 10000000))),
  portfolio = "Market portfolio",
  agnostic_of_TC = rep(c("Agnostic", "Cost-sensitve"), each = length(seq(0.00001, 5000000000, by = 10000000)), times = length(unique(market_return_df$month))),
  scenario = rep(seq(0.00001, 5000000000, by = 10000000), length(unique(market_return_df$month)))
)

# Add columns in order to have same format as all_results
market_portfolio_rows$turnover <- 0
market_portfolio_rows$avg_mktcap_share <- 1

# Append the market portfolio to all_results
breakeven_data <- rbind(all_breakeven_results, market_portfolio_rows)



#Calculate and add long short portfolio
long_short_rows <- breakeven_data |>
  group_by(month, scenario, agnostic_of_TC) |>
  summarize(
    net_return = sum(net_return[portfolio == "Long"]) - sum(net_return[portfolio == "Short"]),
    turnover = (turnover[portfolio=="Long"] + turnover[portfolio=="Short"] )/2,
    avg_mktcap_share = (avg_mktcap_share[portfolio=="Long"] + avg_mktcap_share[portfolio=="Short"] )/2
    
  ) |>
  mutate(portfolio = "Long-Short")

# Append the Long_short rows to the original dataframe
breakeven_data <- bind_rows(breakeven_data, long_short_rows)



# Evaluate the different portfolois for each wealth level
breakeven_results <- data.frame()

for (portfolio in c("Long", "Short", "Long-Short", "Market portfolio")) {
  # loop over each wealth level
  for (scenario in unique(breakeven_data$scenario)) {
    # loop over each value of agnostic
    for (Agnostic_label in c("Agnostic", "Cost-sensitve")) {
      
      # Call the portfolio_evaluation function 
      evaluation_result <- portfolio_evaluation(breakeven_data, portfolio, scenario, Agnostic_label)
      
      # Store the evaluation result
      breakeven_results <- rbind(breakeven_results, evaluation_result)
      
    }
  }
}

#write.csv(breakeven_results, "breakeven_results.csv", row.names = FALSE)

# Choose only Long-Short portfolio
long_short_data <- breakeven_results |>
  filter(portfolio == "Long-Short", scenario <= 5000000000) |>
  mutate(turnover= 100*turnover)



# Convert wealth to mUSD
long_short_data$scenario <- as.numeric(long_short_data$scenario / 1000000)



# Reshape the data for ease of plotting
long_short_data_long <- long_short_data |>
  pivot_longer(cols = c(alpha, annualized_sharpe_ratio   , turnover, annualized_return),
               names_to = "Variable", values_to = "Value")

# Update the names of the variables
long_short_data_long$Variable <- factor(long_short_data_long$Variable, 
                                        levels = c("annualized_return", "alpha", "annualized_sharpe_ratio", "turnover"),
                                        labels = c("Net Return" ,"Alpha", "Sharpe Ratio", "Turnover" ))



# Plot breakeven figure
Plot_3 <- ggplot(data = long_short_data_long, aes(x = scenario, y = Value, linetype = Agnostic_label, color = Agnostic_label)) +
  geom_line(size=1) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +  # Add a horizontal line at y = 0 + 
  facet_grid(facets = vars(Variable), scales = "free_y",) +
  ggtitle("") +
  theme_gray() +
  theme(plot.title = element_text(hjust = 0.5),  axis.line.y = element_line(color = "gray"),
        text = element_text(size = 15), legend.position = "bottom") +
  coord_cartesian(ylim = c(0, NA)) +
  scale_color_manual(values = c("Cost-sensitve" = "black", "Agnostic" = "darkred")) + 
  labs(x = "Portfolio Size (mUSD)", y=NULL, linetype = "", color = "") + 
  geom_hline(data = subset(long_short_data_long, Variable == "Sharpe Ratio"), 
             aes(yintercept = 0.68), linetype = "dotdash", color = "darkblue", size = 0.8) + # Set y-axis limit from 0 to maximum value
  geom_ribbon(data = subset(long_short_data_long, Variable == "Alpha"),
              aes(ymin = alpha_lb, ymax = alpha_ub), alpha = 0.4, fill = "gray", color = NA) +
  geom_ribbon(data = subset(long_short_data_long, Variable == "Sharpe Ratio"),
              aes(ymin = SR_lb, ymax = SR_ub), alpha = 0.4, fill = "gray", color = NA) +
  geom_hline(data = subset(long_short_data_long, Variable == "Net Return"), 
             aes(yintercept = 10.24), linetype = "dotdash", color = "darkblue", size = 0.8)  # Set y-axis limit from 0 to maximum value
  
Plot_3



ggsave("plot_3.png", plot = Plot_3, bg = "white", dpi = 600, width = 12, height = 13)


######################  Characterisitics analysis - (THIS PART OF THE CODE IS SUBJECT TO CODE REPETITON AND IS NOT WELL COMMENTED) ######################



# Set path to data with all the firms characteristics to merge them with the chosen firms 
tidy_finance <- dbConnect(
  SQLite(),
  "tidy_finance_ML.sqlite",
  extended_types = TRUE)

# Collect monthly stock characteristics from 1985 and onwards
stock_characteristics_monthly <- tbl(tidy_finance, "stock_characteristics_monthly") |>
  collect() |> filter(month >= "1985-01-01")




# Select variables to include (all from Gu et al paper)
variables <- stock_characteristics_monthly |> select(c(permno, month, ret_excess, mktcap_lag, sic2, macro_dfy, macro_svar, 
                                                       macro_dp, macro_ep, macro_bm, macro_ntis, macro_tbl, macro_tms,
                                                       characteristic_absacc, characteristic_acc, characteristic_aeavol, characteristic_age, characteristic_agr,
                                                       characteristic_baspread, characteristic_beta, characteristic_betasq, characteristic_bm, characteristic_bm_ia,
                                                       characteristic_cash, characteristic_cashdebt, characteristic_cashpr, characteristic_cfp, characteristic_cfp_ia,
                                                       characteristic_chatoia, characteristic_chcsho, characteristic_chempia, characteristic_chinv, characteristic_chmom,
                                                       characteristic_chpmia, characteristic_chtx, characteristic_cinvest, characteristic_convind, characteristic_currat,
                                                       characteristic_depr, characteristic_dolvol, characteristic_dy,
                                                       characteristic_ear, characteristic_egr, characteristic_ep, characteristic_gma, characteristic_grcapx, 
                                                       characteristic_grltnoa, characteristic_herf, characteristic_hire, characteristic_idiovol, characteristic_ill,
                                                       characteristic_indmom, characteristic_invest, characteristic_lev, characteristic_lgr, characteristic_maxret,
                                                       characteristic_mom12m, characteristic_mom1m, characteristic_mom36m, characteristic_mom6m, characteristic_ms,
                                                       characteristic_mvel1, characteristic_mve_ia, characteristic_nincr, characteristic_operprof, characteristic_orgcap,
                                                       characteristic_pchcapx_ia, characteristic_pchcurrat, characteristic_pchdepr, characteristic_pchgm_pchsale,
                                                       characteristic_pchquick, characteristic_pchsale_pchinvt, characteristic_pchsale_pchrect, characteristic_pchsale_pchxsga,
                                                       characteristic_pchsaleinv, characteristic_pctacc, characteristic_pricedelay, characteristic_ps, characteristic_quick,
                                                       characteristic_rd, characteristic_rd_mve, characteristic_rd_sale, characteristic_realestate, characteristic_retvol,
                                                       characteristic_roaq, characteristic_roavol, characteristic_roeq, characteristic_roic, characteristic_rsup,
                                                       characteristic_salecash, characteristic_saleinv, characteristic_salerec, characteristic_secured, 
                                                       characteristic_securedind, characteristic_sgr, characteristic_sp, characteristic_std_dolvol,
                                                       characteristic_std_turn, characteristic_tang, characteristic_tb,
                                                       characteristic_turn, characteristic_zerotrade, characteristic_stdacc, characteristic_stdcf,
                                                       characteristic_divi, characteristic_divo, characteristic_sin
)

) 


collective_predicted_values$month <- as.Date(collective_predicted_values$month)


######################################## Retrieving agnostic and cost-sensitive portfolio companies ################################################


# Set cost-sensitve scenario
scenarios <- list(scenario_5 = list(wealth_level = 100000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50"),
                  scenario_3 = list(wealth_level = 0.0000001, Agnostic_of_TC = TRUE, equal_weighting = TRUE, Sort_type = "Top/buttom 50"))


# Find firms in cost-sensitve and agostic porfolios
all_results <- data.frame()
all_chosen_weights <- list()

for (portfolio_type in c("Long", "Short")) {
  for (scenario_name in names(scenarios)) {
    
    # Set scenario name
    scenario <- scenarios[[scenario_name]]
    
    # Set wealth level 
    wealth <- scenario$wealth_level
    
    # Set initial old weights data 
    Old_weights_data <- merged_data|>
      filter(month == "2005-01-01")|>
      select(permno)|>
      mutate(Old_weights = 0)
    
    # Create an empty list to store chosen_weights datasets for each scenario
    chosen_weights_list <- list()
    
    # Loop over all the months
    for (current_month in seq(as.Date("2005-01-01"), as.Date("2021-11-01"), by = "months")) {
      
      # Choose weights
      chosen_weights <- portfolio_weights(data = merged_data,
                                          current_month = current_month,
                                          wealth = wealth,
                                          Old_weights_data = Old_weights_data,
                                          equal_weighting = scenario$equal_weighting,
                                          portfolio_type = portfolio_type,
                                          Agnostic_of_TC = scenario$Agnostic_of_TC,
                                          Sort_type = scenario$Sort_type)
      
      # Store chosen_weights dataset for each month
      chosen_weights_list[[as.character(current_month)]] <- chosen_weights
      
      # Store old weights
      Old_weights_data <- chosen_weights|>
        select(permno, Old_weights)
      
      # Collect market return and risk free
      market_ret <- chosen_weights$market_ret[which(!is.na(chosen_weights$market_ret))[1]]
      risk_free <- chosen_weights$rf[which(!is.na(chosen_weights$rf))[1]]
      
      # Update wealth level for next month
      wealth <- wealth * (1 + market_ret + risk_free)
    }
    
    # Combine chosen_weights datasets for each month into a single dataframe
    chosen_weights_combined <- bind_rows(chosen_weights_list, .id = "Month")
    
    # Add a column for scenario name
    chosen_weights_combined$Scenario <- scenario_name
    
    # Add a column for portfolio type
    chosen_weights_combined$Portfolio_Type <- portfolio_type
    
    # Store combined chosen_weights dataset in all_chosen_weights list
    all_chosen_weights[[paste(scenario_name, portfolio_type, sep = "_")]] <- chosen_weights_combined
  }
}

# Bind all chosen_weights datasets into a single dataframe
all_chosen_weights_df <- do.call(rbind, all_chosen_weights)


#Choose the long portfolio
Long_Portfolio_firms <- all_chosen_weights_df |>
  filter(portfolio=="Long") |>   #Change to Short here to find entropy levels of characteristics of firms in short portfolios
  select(month, permno, mktcap_lag, ret_excess, pred, portfolio, Scenario) 



### Select characteristics and for all firms ###
characteristic_list <- c(
                         "characteristic_absacc", "characteristic_acc", "characteristic_aeavol", "characteristic_age", "characteristic_agr",
                         "characteristic_baspread", "characteristic_beta", "characteristic_betasq", "characteristic_bm", "characteristic_bm_ia",
                         "characteristic_cash", "characteristic_cashdebt", "characteristic_cashpr", "characteristic_cfp", "characteristic_cfp_ia",
                         "characteristic_chatoia", "characteristic_chcsho", "characteristic_chempia", "characteristic_chinv", "characteristic_chmom",
                         "characteristic_chpmia", "characteristic_chtx", "characteristic_cinvest", "characteristic_convind", "characteristic_currat",
                         "characteristic_depr", "characteristic_dolvol", "characteristic_dy",
                         "characteristic_ear", "characteristic_egr", "characteristic_ep", "characteristic_gma", "characteristic_grcapx",
                         "characteristic_grltnoa", "characteristic_herf", "characteristic_hire", "characteristic_idiovol", "characteristic_ill",
                         "characteristic_indmom", "characteristic_invest", "characteristic_lev", "characteristic_lgr", "characteristic_maxret",
                         "characteristic_mom12m", "characteristic_mom1m", "characteristic_mom36m", "characteristic_mom6m", "characteristic_ms",
                         "characteristic_mvel1", "characteristic_mve_ia", "characteristic_nincr", "characteristic_operprof", "characteristic_orgcap",
                         "characteristic_pchcapx_ia", "characteristic_pchcurrat", "characteristic_pchdepr", "characteristic_pchgm_pchsale",
                         "characteristic_pchquick", "characteristic_pchsale_pchinvt", "characteristic_pchsale_pchrect", "characteristic_pchsale_pchxsga",
                         "characteristic_pchsaleinv", "characteristic_pctacc", "characteristic_pricedelay", "characteristic_ps", "characteristic_quick",
                         "characteristic_rd", "characteristic_rd_mve", "characteristic_rd_sale", "characteristic_realestate", "characteristic_retvol",
                         "characteristic_roaq", "characteristic_roavol", "characteristic_roeq", "characteristic_roic", "characteristic_rsup",
                         "characteristic_salecash", "characteristic_saleinv", "characteristic_salerec", "characteristic_secured",
                         "characteristic_securedind", "characteristic_sgr", "characteristic_sp", "characteristic_std_dolvol",
                         "characteristic_std_turn", "characteristic_tang", "characteristic_tb",
                         "characteristic_turn", "characteristic_zerotrade", "characteristic_stdacc", "characteristic_stdcf",
                         "characteristic_divi", "characteristic_divo", "characteristic_sin")



#Choose relevant characteristics
characteristic <- variables |>
  select(month, permno, all_of(characteristic_list))


#Create data set for distrubution of characteristic for all firms
all_firms <- left_join(collective_predicted_values, characteristic, by = c("month","permno")) |>
  mutate(Scenario = "All_firms") 

# Merge the chosen firms and the characterisics
Long_Portfolio_firms <- left_join(Long_Portfolio_firms, characteristic, by = c("month","permno")) |>
  mutate(Scenario = case_when(
    Scenario == "scenario_3" ~ "Agnostic")) |>
  drop_na()




# Initialize an empty vector to store KL divergence values
kl_values <- numeric(length(characteristic_list))

# Loop over each characteristic
for (char in characteristic_list) {
  # Extract the data for the characteristic
  x <- Long_Portfolio_firms[[char]]
  y <- all_firms[[char]]
  
  # Set the range and bin width
  range_min <- -1
  range_max <- 1
  bin_width <- 0.2
  
  # Calculate density for x and y
  density_x <- density(x, from = range_min, to = range_max, bw = bin_width)
  density_y <- density(y, from = range_min, to = range_max, bw = bin_width)
  
  # Normalize the densities
  P <- density_x$y / sum(density_x$y)
  Q <- density_y$y / sum(density_y$y)
  
  # Calculate KL divergence
  z <- rbind(P, Q)
  kl_values[char] <- KL(z)
}

# Combine characteristic names and KL divergence values
kl_results <- data.frame(characteristic = characteristic_list, KL_divergence = kl_values)


kl_results_sorted <- kl_results |>
  arrange(desc(KL_divergence)) 

# View the sorted results
print(kl_results_sorted)

#write.csv(kl_results_sorted, "Entropy.csv", row.names = FALSE)

######################### CREATING FIGURES FOR MOST IMPORTANT CHARACTERISTICS #####################################################

characteristic_list <- c("characteristic_mvel1", "characteristic_mom1m", "characteristic_ill", 
                         "characteristic_retvol", "characteristic_mom12m", "characteristic_ep",
                         "characteristic_cashdebt", "characteristic_cfp", "characteristic_roic")

characteristic <- variables |>
  select(month, permno, all_of(characteristic_list))


All_Portfolio_firms <- all_chosen_weights_df |>
  filter(portfolio %in% c("Long", "Short")) |>
    mutate(Scenario = case_when(
      Scenario == "scenario_3" ~ "Agnostic",
      Scenario == "scenario_5" ~ "Cost-sensitve")) |>
  drop_na() |>
  select(month, permno, mktcap_lag, ret_excess, pred, portfolio, Scenario) 


Portfolio_firms_merge <- left_join(All_Portfolio_firms, characteristic, by = c("month","permno"))


Long_portfolio_firms_merge <- Portfolio_firms_merge |>
  filter(portfolio == "Long")

Short_portfolio_firms_merge <- Portfolio_firms_merge |>
  filter(portfolio == "Short")

#Renaming
long_portfolio_firms_long <- Long_portfolio_firms_merge |>
  pivot_longer(cols = all_of(characteristic_list),
               names_to = "characteristic",
               values_to = "value") |>
  mutate(characteristic = gsub("^characteristic_", "", characteristic))

short_portfolio_firms_long <- Short_portfolio_firms_merge |>
  pivot_longer(cols = all_of(characteristic_list),
               names_to = "characteristic",
               values_to = "value") |>
  mutate(characteristic = gsub("^characteristic_", "", characteristic))


long_portfolio_firms_long <- long_portfolio_firms_long|>
  mutate(characteristic = recode(characteristic,
                                  mvel1 = "Size",
                                  mom1m = "1-Month Momentum",
                                  ill = "Illiquidity",
                                  retvol = "Return Volatility",
                                  mom12m = "12-Month Momentum",
                                  ep = "Earnings/Price",
                                  cashdebt = "Cash Flow/Debt",
                                  cfp = "Cash Flow/Price",
                                  roic = "Return on Invested Capital"
  ))

short_portfolio_firms_long <- short_portfolio_firms_long|>
  mutate(characteristic = recode(characteristic,
                                 mvel1 = "Size",
                                 mom1m = "1-Month Momentum",
                                 ill = "Illiquidity",
                                 retvol = "Return Volatility",
                                 mom12m = "12-Month Momentum",
                                 ep = "Earnings/Price",
                                 cashdebt = "Cash Flow/Debt",
                                 cfp = "Cash Flow/Price",
                                 roic = "Return on Invested Capital"
  ))



###################### PLOTTING CHARACTERISTICS #################################################################################


desired_order <- c("Size", "1-Month Momentum", "Illiquidity", "Return Volatility", 
                   "12-Month Momentum", "Earnings/Price", "Cash Flow/Debt", "Cash Flow/Price", "Return on Invested Capital")



# Convert the characteristic variable to a factor with desired order
long_portfolio_firms_long$characteristic <- factor(long_portfolio_firms_long$characteristic, levels = desired_order)
short_portfolio_firms_long$characteristic <- factor(short_portfolio_firms_long$characteristic, levels = desired_order)



p <- ggplot(long_portfolio_firms_long, aes(x=value, fill = Scenario, color = Scenario)) + 
  geom_density(alpha=0.5, size = 1) +
  facet_wrap(characteristic ~ ., scales = "free_y", ncol = 3) +
  geom_density(data=short_portfolio_firms_long, alpha = 0.2,size = 1,  aes(y = -..density..))  +
  labs(x = "", y=NULL, fill = "", color = "") +
  scale_fill_manual(values=c("darkgray", "#E69F00")) +
  scale_color_manual(values=c("darkgray", "#E69F00")) + 
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 1,5) + 
  theme(legend.position = "bottom", text = element_text(size = 12))  # Place legend at the bottom

p


ggsave("characteristics.png"  , plot = p, bg = "white", dpi = 600, width = 11.5, height = 10)



############## Portfolio characteristics for cost-sensitive portfolio for different wealth levels ################

### Retrieving firms in cost-sensitve portfolios for different wealth levels ###

all_results <- data.frame()
all_chosen_weights <- list()

for (portfolio_type in c("Long")) {
  for (wealth_level in c(0.000001, 100000000, 1000000000, 3000000000, 5000000000)) {
  
    
    # Set wealth level 
    wealth <- wealth_level
    
    # Set initial old weights data 
    Old_weights_data <- merged_data|>
      filter(month == "2005-01-01")|>
      select(permno)|>
      mutate(Old_weights = 0)
    
    # Create an empty list to store chosen_weights datasets for each scenario
    chosen_weights_list <- list()
    
    # Loop over all the months
    for (current_month in seq(as.Date("2005-01-01"), as.Date("2021-11-01"), by = "months")) {
      
      # Choose weights
      chosen_weights <- portfolio_weights(data = merged_data,
                                          current_month = current_month,
                                          wealth = wealth,
                                          Old_weights_data = Old_weights_data,
                                          equal_weighting = TRUE,
                                          portfolio_type = portfolio_type,
                                          Agnostic_of_TC = FALSE,
                                          Sort_type = "Top/buttom 50")
      
      # Store chosen_weights dataset for each month
      chosen_weights_list[[as.character(current_month)]] <- chosen_weights
      
      # Store old weights
      Old_weights_data <- chosen_weights|>
        select(permno, Old_weights)
      
      # Collect market return and risk free
      market_ret <- chosen_weights$market_ret[which(!is.na(chosen_weights$market_ret))[1]]
      risk_free <- chosen_weights$rf[which(!is.na(chosen_weights$rf))[1]]
      
      # Update wealth level for next month
      wealth <- wealth * (1 + market_ret + risk_free)
    }
    
    # Combine chosen_weights datasets for each month into a single dataframe
    chosen_weights_combined <- bind_rows(chosen_weights_list, .id = "Month")
    
    # Add a column for scenario name
    chosen_weights_combined$Wealth <- wealth_level
    
    # Add a column for portfolio type
    chosen_weights_combined$Portfolio_Type <- portfolio_type
    
    # Store combined chosen_weights dataset in all_chosen_weights list
    all_chosen_weights[[paste(wealth_level, portfolio_type, sep = "_")]] <- chosen_weights_combined
  }
}

#Bind all chosen_weights datasets into a single dataframe
all_chosen_weights_df <- do.call(rbind, all_chosen_weights)

all_chosen_weights_df <- all_chosen_weights_df|>
  mutate(Wealth = case_when(
    Wealth == 0.000001 ~ "0 (Agnostic)",
    Wealth == 100000000 ~ "100",
    Wealth == 1000000000 ~ "1000",
    Wealth == 3000000000 ~ "3000",
    Wealth == 5000000000 ~ "5000",
    TRUE ~ as.character(Wealth)
  ))

#Choose characteristics to look at
characteristic <- variables |>
  select(month, permno, all_of(characteristic_list))

#Choose firms in cost-sensitive long portfolio
All_Portfolio_firms <- all_chosen_weights_df |>
  filter(portfolio %in% c("Long")) |>
  drop_na() |>
  select(month, permno, mktcap_lag, ret_excess, pred, portfolio, Wealth) 

# merge firms and characteristics
Portfolio_firms_merge <- left_join(All_Portfolio_firms, characteristic, by = c("month","permno"))


#Renaming
long_portfolio_firms_long <- Portfolio_firms_merge |>
  pivot_longer(cols = all_of(characteristic_list),
               names_to = "characteristic",
               values_to = "value") |>
  mutate(characteristic = gsub("^characteristic_", "", characteristic))


long_portfolio_firms_long <- long_portfolio_firms_long|>
  mutate(characteristic = recode(characteristic,
                                 mvel1 = "Size",
                                 mom1m = "1-Month Momentum",
                                 ill = "Illiquidity",
                                 retvol = "Return Volatility",
                                 mom12m = "12-Month Momentum",
                                 ep = "Earnings/Price",
                                 cashdebt = "Cash Flow/Debt",
                                 cfp = "Cash Flow/Price",
                                 roic = "Return on Invested Capital"
  ))





#Plotting
long_portfolio_firms_long$Wealth <- factor(long_portfolio_firms_long$Wealth, levels = c("0 (Agnostic)", "100", "1000", "3000", "5000"))


# Convert the characteristic variable to a factor with desired order
long_portfolio_firms_long$characteristic <- factor(long_portfolio_firms_long$characteristic, levels = desired_order)



p <- ggplot(long_portfolio_firms_long, aes(x=value, y=Wealth, group = Wealth)) + 
  geom_density_ridges(alpha=0.2) +
  facet_wrap(characteristic ~ ., ncol = 3) +
  labs(x = "", y="Portfolio Size (mUSD)", fill = "", color = "") +
  theme(legend.position = "bottom", text = element_text(size = 12))  # Place legend at the bottom

p


ggsave("characteristics_Wealth.png"  , plot = p, bg = "white", dpi = 600, width = 11.5, height = 10)



################################# FAMA FRENCH RISK FACTOR RISK ANALYSIS #############################################


#Fetch data
factors_ff5_monthly_raw <- download_french_data("Fama/French 5 Factors (2x3)")

# Manipulate data
factors_ff5_monthly <- factors_ff5_monthly_raw$subsets$data[[1]] |>
  mutate(
    month = floor_date(ymd(str_c(date, "01")), "month"),
    across(c(RF, `Mkt-RF`, SMB, HML, RMW, CMA), ~as.numeric(.) / 100),
    .keep = "none"
  ) |>
  rename_with(str_to_lower) |>
  rename(mkt_excess = `mkt-rf`) |> 
  filter(month >= "1985-01-01" & month <= "2022-01-01")


momentum <- download_french_data("Momentum Factor (Mom)")

momentum <- momentum$subsets$data[[1]] |>
  mutate(
    month = floor_date(ymd(str_c(date, "01")), "month"),
    MOM = Mom/100) |>
  select(-date, -Mom) |>
  filter(month >= "1985-01-01" & month <= "2022-01-01")
  
FF_data <- merge(factors_ff5_monthly, momentum, by = c("month"))



# Creating different scenarios/specifications
scenarios <- list(
  
  scenario_1 = list(wealth_level = 0.00001, Agnostic_of_TC = FALSE, equal_weighting = FALSE, Sort_type = "Top/buttom 50", gross = FALSE),
  
  scenario_2 = list(wealth_level = 100000000, Agnostic_of_TC = FALSE, equal_weighting = FALSE, Sort_type = "Top/buttom 50", gross = FALSE),
  
  scenario_3 = list(wealth_level = 1000000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = FALSE),
  
  scenario_4 = list(wealth_level = 3000000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = FALSE),
  
  scenario_5 = list(wealth_level = 5000000000, Agnostic_of_TC = FALSE, equal_weighting = TRUE, Sort_type = "Top/buttom 50", gross = FALSE)

)


# Portfolio backtesting: Looping over the different scenarios and portfolio types
all_results <- data.frame()

for (portfolio_type in c("Long", "Short")) {
  for (scenario_name in names(scenarios)) {
    
    #Set scenario name
    scenario <- scenarios[[scenario_name]]
    
    #Set wealth level 
    wealth <- scenario$wealth_level
    
    #Set initial old weights data 
    Old_weights_data <- merged_data |>
      filter(month == "2005-01-01") |>
      select(permno) |>
      mutate(Old_weights = 0)
    
    
    # Loop over all the months
    for (current_month in seq(as.Date("2005-01-01"), as.Date("2021-11-01"), by = "months")) {
      
      
      # Choose weights
      chosen_weights <- portfolio_weights(data = merged_data,
                                          current_month = current_month,
                                          wealth = wealth,
                                          Old_weights_data = Old_weights_data,
                                          equal_weighting = scenario$equal_weighting,
                                          portfolio_type = portfolio_type,
                                          Agnostic_of_TC = scenario$Agnostic_of_TC,
                                          Sort_type = scenario$Sort_type)
      
      
      # Storing monthly performance of chosen portfolio
      results <- store_monthly_portfolio(data = chosen_weights, 
                                         portfolio_type = portfolio_type,
                                         scenario = scenario_name,    
                                         Agnostic_of_TC = scenario$Agnostic_of_TC,
                                         current_month = current_month,
                                         gross = scenario$gross)
      
      # Bind results
      all_results <- bind_rows(all_results, results)
      
      # Store old weights
      Old_weights_data <- chosen_weights |>
        select (permno, Old_weights)
      
      # Collect market return and risk free
      market_ret <- chosen_weights$market_ret[which(!is.na(chosen_weights$market_ret))[1]]
      risk_free <- chosen_weights$rf[which(!is.na(chosen_weights$rf))[1]]
      
      # Update wealth level for next month
      wealth <- wealth * (1 + market_ret + risk_free)
    }
  }
}



# Retrieve market portfolio for each of the scenarios in all_results
market_portfolio_rows <- data.frame(
  month = unique(market_return_df$month),  
  net_return = market_return_df$market_ret[match(unique(market_return_df$month), market_return_df$month)],
  portfolio = "Market portfolio",
  scenario = rep(c("scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"), each = length(unique(market_return_df$month))))



# Calculate long short portfolio
long_short_rows <- all_results |>
  group_by(month, scenario, agnostic_of_TC) |>
  summarize(
    net_return = sum(net_return[portfolio == "Long"]) - sum(net_return[portfolio == "Short"]),
    turnover = (turnover[portfolio=="Long"] + turnover[portfolio=="Short"] )/2,
    avg_mktcap_share = (avg_mktcap_share[portfolio=="Long"] + avg_mktcap_share[portfolio=="Short"] )/2
  ) |>
  mutate(portfolio = "Long-Short")   


# Append the Long_short portfolio to all_results
merged_all_results <- bind_rows(all_results, long_short_rows) |>
  filter(portfolio == "Long-Short")


# Merge market portfolio returns with the FF factor portfolios
FF_data <- merge(FF_data, market_portfolio_rows |> filter(scenario == "scenario_1") |> select(net_return, month), by = c("month")) |>
  mutate(mkt = net_return) |>
  select(-net_return, -mkt_excess, -rf)

data_for_reg <- FF_data

# List of scenarios
scenarios <- paste0("scenario_", 1:5)

# Iterate over each scenario and merge FF factor portfolios with return of cost-sensitive portfolio for each of the wealth levels
for (scenario in scenarios) {
  chosen_scenario <- merged_all_results |>
    filter(scenario == !!scenario) |>
    select(net_return, month)
  
  # Merge with data_for_reg and rename the net_return column
  data_for_reg <- merge(data_for_reg, chosen_scenario, by = "month", all.x = TRUE)|>
    rename(!!scenario := net_return)
}


# Perform regressions
# Initialize a list to store regression results
regression_results <- list()

# Loop through each scenario and perform the regression
for (scenario in scenarios) {
  # Define the regression formula
  formula <- as.formula(paste(scenario, "~ smb + hml + rmw + cma + mkt + MOM"))
  
  # Perform the regression
  model <- lm(formula, data = data_for_reg)
  
  # Get Newey-West standard errors using coeftest
  coef_test <- coeftest(model, vcov = NeweyWest(model))
  
  # Extract coefficients and Newey-West standard errors from coef_test
  coefficients <- coef_test[, "Estimate"]
  nw_se <- coef_test[, "Std. Error"]
  
  # Store results in a named list
  regression_results[[scenario]] <- list(
    alpha_FF = 12 * 100 * coefficients["(Intercept)"],
    beta_smb = coefficients["smb"],
    beta_hml = coefficients["hml"],
    beta_rmw = coefficients["rmw"],
    beta_cma = coefficients["cma"],
    beta_mkt = coefficients["mkt"],
    beta_MOM = coefficients["MOM"],
    alpha_FF_se = sqrt(12) * 100 * nw_se["(Intercept)"],
    beta_smb_se = nw_se["smb"],
    beta_hml_se = nw_se["hml"],
    beta_rmw_se = nw_se["rmw"],
    beta_cma_se = nw_se["cma"],
    beta_mkt_se = nw_se["mkt"],
    beta_MOM_se = nw_se["MOM"],
    r_squared = summary(model)$r.squared
  )
}

# Convert the results to a data frame for easier viewing
results_df <- do.call(rbind, lapply(regression_results, as.data.frame))
results_df <- cbind(scenario = rownames(results_df), results_df)
rownames(results_df) <- NULL


# Creating function to check significance and format coefficients
format_coeff <- function(coeff, se) {
  t_stat <- abs(coeff / se)
  formatted_coeff <- format(round(coeff, 2), nsmall = 2)
  if (t_stat > 1.96) {
    return(paste0(formatted_coeff, "*"))
  } else {
    return(formatted_coeff)
  }
}

# Create a data frame for LaTeX table
latex_table <- data.frame(
  scenario = c("scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"),
  alpha_FF = mapply(format_coeff, results_df$alpha_FF, results_df$alpha_FF_se),
  beta_smb = mapply(format_coeff, results_df$beta_smb, results_df$beta_smb_se),
  beta_hml = mapply(format_coeff, results_df$beta_hml, results_df$beta_hml_se),
  beta_rmw = mapply(format_coeff, results_df$beta_rmw, results_df$beta_rmw_se),
  beta_cma = mapply(format_coeff, results_df$beta_cma, results_df$beta_cma_se),
  beta_mkt = mapply(format_coeff, results_df$beta_mkt, results_df$beta_mkt_se),
  beta_MOM = mapply(format_coeff, results_df$beta_MOM, results_df$beta_MOM_se),
  r_squared = format(round(results_df$r_squared, 2), nsmall = 2)
)

# Generate LaTeX table code
latex_code <- "
\\begin{table}[ht]
\\centering
\\begin{tabular}{lcccccccc}
\\hline
Scenario & $\\alpha_{FF}$ & $\\beta_{smb}$ & $\\beta_{hml}$ & $\\beta_{rmw}$ & $\\beta_{cma}$ & $\\beta_{mkt}$ & $\\beta_{MOM}$ & $R^2$ \\\\
\\hline
"
for (i in 1:nrow(latex_table)) {
  latex_code <- paste0(latex_code, latex_table[i, "scenario"], " & ",
                       latex_table[i, "alpha_FF"], " & ",
                       latex_table[i, "beta_smb"], " & ",
                       latex_table[i, "beta_hml"], " & ",
                       latex_table[i, "beta_rmw"], " & ",
                       latex_table[i, "beta_cma"], " & ",
                       latex_table[i, "beta_mkt"], " & ",
                       latex_table[i, "beta_MOM"], " & ",
                       latex_table[i, "r_squared"], " \\\\\n")
}

latex_code <- paste0(latex_code, "\\hline\n\\end{tabular}\n\\caption{Regression Results with Significance Indicated by *}\n\\end{table}")

# Print LaTeX code
cat(latex_code)
