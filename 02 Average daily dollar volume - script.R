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

                                                  ### DISCLAIMER: This script creates the measure of daily dollar volume used in our paper based on daily CRSP data
                                                  ### The output of the script is used as input in the Backtesting script
                                                  ### Set working directory to where the daily CRSP data is stored (sfz_dp_dly.rds)
                                                  ### Choose where to store the results in line 71



data <- readRDS("sfz_dp_dly.rds") |>
  filter(CALDT >= "2020-01-01") |>
  rename(permno = KYPERMNO)


# Add month format
data$month <- as.Date(format(data$CALDT, "%Y-%m-01"))  

# Convert PRC to absolute values (negative sign indicates that it is a bid/ask average and not an actual closing price. 
#The negative sign is a symbol and the value of the bid/ask average is not negative. See CRSP documentation for details)
data$PRC <- abs(data$PRC)


#Create dataframe to store results  
final_result_df <- data.frame(permno = character(),
                              avg_dolvol = numeric(),
                              stringsAsFactors = FALSE)  

#Calculate average daily dollar volume across the past 6 months
for (i in seq(as.Date("1985-01-01"), as.Date("2021-11-01"), by = "months")) {
  # Filter data for the six prior months
  subset_data <- data |>
    filter(month > (as.Date(i) - months(6)) & month <= as.Date(i))
  
  # Calculate dolvol
  avg_dolvol <- subset_data |>
    mutate(dolvol = PRC * VOL) |>
    select(dolvol, permno) |>
    group_by(permno) |>
    summarize(avg_dolvol = mean(dolvol, na.rm = TRUE))
  
  # Add Month column
  avg_dolvol$month <- as.Date(i)
  avg_dolvol$permno <- as.character(avg_dolvol$permno)
  
  
  # Bind results to the final data frame
  final_result_df <- bind_rows(final_result_df, avg_dolvol)
}


final_result_df <- final_result_df |>
  arrange(month, permno)


write.csv(final_result_df, "C:/Users/bjark/Desktop/Speciale/R results/CRSPdolvol_v2.csv", row.names=FALSE)







