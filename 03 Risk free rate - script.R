library(tidyverse)
library(RSQLite)
library(lubridate)
library(scales)
library(slider)
library(furrr)
library(tidyr)
library(ggplot2)
library(dplyr)


                                      #### DISCLAIMER: This script fetches the monthly risk free rate from CRSP used to adjust return to create excess returns ###
                                      #### The output is used as input in the Backtesting script. Set working folder to where the tidy_finance.sqlite is located ###
                                      #### Specify in line 33 where to store the results ###


#Load data
tidy_finance <- dbConnect(
  SQLite(), "tidy_finance.sqlite",
  extended_types = TRUE
)

crsp_monthly <- tbl(tidy_finance, "crsp_monthly") |>
  collect()


crsp_ret <- crsp_monthly |>
  mutate(ret_excess_check = ret_excess, rf = ret-ret_excess) |>
  select(permno, month, ret, rf, ret_excess_check ) |> 
  filter(month >= "2005-01-01")


write.csv(crsp_ret, "C:/Users/bjark/Desktop/Speciale/R results/rfdata.csv", row.names=FALSE)