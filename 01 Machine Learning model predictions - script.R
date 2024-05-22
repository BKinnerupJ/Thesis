library(RSQLite)
library(tidyverse)
library(scales)
library(furrr)
library(broom)
library(tidymodels)
library(glmnet)
library(timetk)
library(keras)
library(ranger)
library(hardhat)
library(sandwich)
library(patchwork)
library(lubridate)


                                                                         #### DISCLAIMER ###
                        ### This code computes trains our ML model and genererates monthly return predictions using CRSP data. 
                        ### The data can be created by having acces to CRSP and by following: https://voigtstefan.quarto.pub/course-exercises/gu_kelly_xiu.html 
                        ### On a machine with a processor with a 2,4GHZ processor (20 logical cores) and 32GB of RAM, this code takes approximately 20 hours to run 
                        ### The ML model is very memory demanding - removing variables and do calculations for fewer years will reduce memory usage 
                        ### Set working directory to the folder where your tidy_finance_ML.splite file is located 
                        ### Set where to store the yearly predictions on your machine in line 335 (to reduce memory usage, a csv files with results
                        ### is stored locally after each year of monthly return predictions)
                        ### The code requires an installation of tensorflow


# Timing how long the code takes to run
Starttime <- Sys.time()

# Set path to data
tidy_finance <- dbConnect(
  SQLite(),
  "tidy_finance_ML.sqlite",
  extended_types = TRUE)

# Collect monthly stock characteristics from 1985 and onwards
stock_characteristics_monthly <- tbl(tidy_finance, "stock_characteristics_monthly") |>
  collect() |> filter(month >= "1985-01-01", month <= "2005-01-01" )



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

) |> drop_na() |> arrange(month)




# Forcing R to read sic2 (sector) as character instead of a numeric variable in order to convert to dummies
variables$sic2 <- as.character(variables$sic2)

# Creating recipe for dummies for each sic and interaction terms between macro variables and stock characteristics
rec <- recipe(ret_excess ~ ., data = variables )|>
  update_role(permno, month, mktcap_lag, new_role = "id") |>
  step_interact(terms = ~ contains("characteristic"):contains("macro")) |>
  step_dummy(sic2, one_hot = TRUE) 

# Clear PC memory
gc()

# Apply the the recipe to the data
data_prep <- prep(rec, variables)
data_bake <- bake(data_prep, new_data = variables)


# Removing data no longer needed
rm(data_prep)
rm(stock_characteristics_monthly)
rm(variables)
rm(rec)
rm(tidy_finance)



######################################### MODEL TUNING ################################################
# The following loop trains the model on data from 1985 to 1996, and predict one-shot returns for the entire period 1997-2004.
# It does this for two different degrees of l1 penalization¨
# The model delieveing the lowest mean squarred prediction error is then found

# Set first date of training
start_date <- as.Date("1985-01-01")

# Set last date of training
end_date <- as.Date("1997-01-01")
result_list <- list()


# loop through each penalization
for (penalization in c(0.00001 ,0.001)) {
  
  # Set input shape for DNN
  inputshape <- ncol(data_bake) - 4
  
  # Create dataframe to store results
  average_predictions <- numeric(length((data_bake |> filter(month >= end_date & month < end_date + years(8)))$ret_excess))
  
  # loop through 10 different seeds (ensemble)
  for (seed in 1:10) {
    
    # Set seed for reproducibility
    tensorflow::set_random_seed(seed)
    
    model <- keras_model_sequential() |>
      layer_flatten(input_shape = inputshape) |>
      layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l1(penalization)) |>
      layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l1(penalization)) |>
      layer_dense(units = 8, activation = "relu", kernel_regularizer = regularizer_l1(penalization)) |>
      layer_dense(units = 4, activation = "relu", kernel_regularizer = regularizer_l1(penalization)) |>
      layer_dense(1) |>
      compile(
        loss = "mse", optimizer = optimizer_adam(learning_rate = 0.01))
    
    # Train the model on our training data
    model |>
      fit(
        x = data_bake |> filter(month >= start_date & month < end_date) |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix(),
        y = data_bake |> filter(month >= start_date & month < end_date) |> pull(ret_excess),
        validation_data = list(data_bake |> filter(month >= end_date & month < end_date + years(8)) |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix(),
                               data_bake |> filter(month >= end_date & month < end_date + years(8)) |> pull(ret_excess)),
        epochs = 100, verbose = TRUE, batch_size = 5000, callbacks = callback_early_stopping(monitor = "val_loss", patience = 5)
      )
    
    # Clear unused memory
    gc()
    
    # Make out-of-sample predictions with the model
    predicted_values <- model |>
      predict((data_bake) |> filter(month >= end_date & month < end_date + years(8)) |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix()) |>
      as.vector() |>
      tibble(pred = _) |>
      bind_cols(data_bake |> filter(month >= end_date & month < end_date + years(8))) |> select(pred, ret_excess)
    
    gc()
    
    # Add the predictions to the average_predictions (sum up predictions)
    if (is.null(average_predictions)) {
      average_predictions <- predicted_values$pred
    } else {
      average_predictions <- average_predictions + predicted_values$pred
    }
    
    ret_excess <- predicted_values$ret_excess
    
    # Clean up
    rm(model, predicted_values)
    gc()
    
  }
  
  
  # Calculate the average prediction across the 10 iterations (10 seeds)
  average_predictions <- average_predictions / 10
  
  # Store average predictions along with penalization and learning rate in the list
  result_list <- append(result_list, list(data.frame(penalization = penalization,
                                                     prediction = average_predictions,
                                                     ret_excess = ret_excess
  )))
  
  gc()
  
  
}


# Combine dataframes in the list into a single dataframe
result_df <- do.call(rbind, result_list)
write.csv(result_df, "TUNING.csv", row.names = FALSE)


## Evaluating predictive performance for the DNN specifications
# Create an empty data frame to store the results
MSPE_results <- data.frame(penalization = numeric(),
                           MSPE = numeric()
)


# loop through each penalization
for (penalization_ in c(0.00001, 0.001)) {
  
  
  # Calculate MSPE for all firms
  MSPE_data <- result_df |>
    filter(penalization == penalization_) |>
    mutate(SPE = (ret_excess - prediction)^2)
  
  MSPE <- mean(MSPE_data$SPE)
  
  # Append the results to the data frame
  MSPE_results <- rbind(MSPE_results,
                        data.frame(penalization = as.character(penalization_),
                                   MSPE = MSPE
                        ))
  
}



# Plotting the results of the tuning
MSPE_results <- MSPE_results %>%
  mutate(penalization = ifelse(penalization == "0.001", "1e-03", "1e-05"))


g <- ggplot(data = MSPE_results, aes(x = factor(penalization, levels = c("1e-05", "1e-03")), y = MSPE)) +
  geom_bar(stat="identity", width = 0.5) +
  theme_gray() +
  ggtitle("") +
  labs(x = "L1 penalization", y = "MSPE") +
  theme(plot.title = element_text(hjust = 0.5), axis.line.y = element_line(color = "gray"),
        text = element_text(size = 15)) +
  coord_cartesian(ylim = c(0.050, NA)) 

g

ggsave(filename = "Tuning.png", plot = g, dpi = 600, bg = "white")


######################################### OUT OF SAMPLE PREDICTIONS ################################################
# The following loop trains the model on data from 1985 to 1996, using 1997-2004 as validation set for early stopping
# Then it predicts monthly returns for 2005. The training window is then expanded by one year while the validation set is moved 1 year forward
# Based on this, monthly returns are predicted for 2006
# This process continues until predictions are made for 2021
# To reduce system memory requirements, the predictions are saved on the harddrive after each year and then removed from the environment

#Looping through years (1 for 2005, 17 for 2021)
for (i in 17:17) {

  # Set first date of training
  start_date <- as.Date("1985-01-01")

  # Set last date of training
  end_date <- as.Date("1997-01-01") + years(i - 1)

  # Set input shape for DNN
  inputshape <- ncol(data_bake) -4

  #Create dataframe to store results
  average_predictions <- numeric(length((data_bake |> filter(month >= end_date + years(8) & month < end_date + years(9)))$ret_excess))

  #Clear unused memory
  gc()

  # Loop through 10 different seeds (ensemble)
  for (seed in 1:10) {
    tensorflow::set_random_seed(seed)

    # Set specification for DNN model
    model <- keras_model_sequential() |>
      layer_flatten(input_shape = inputshape) |>
      layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l1(0.00001)) |>
      layer_batch_normalization() |>
      layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l1(0.00001)) |>
      layer_batch_normalization() |>
      layer_dense(units = 8, activation = "relu", kernel_regularizer = regularizer_l1(0.00001)) |>
      layer_batch_normalization() |>
      layer_dense(1) |>
      compile(
        loss = "mse", optimizer = optimizer_adam(learning_rate = 0.001))


    # Train the model on our training data
    model |>
      fit(
        x = data_bake |> filter(month >= start_date & month < end_date) |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix(),
        y = data_bake |> filter(month >= start_date & month < end_date) |> pull(ret_excess),
        validation_data = list(data_bake |> filter(month >= end_date & month < end_date + years(8))  |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix(),
                               data_bake |> filter(month >= end_date & month < end_date + years(8)) |> pull(ret_excess)),
        epochs = 100, verbose = TRUE, batch_size = 5000, callbacks = callback_early_stopping(monitor = "val_loss", patience = 5)
      )

    # Clear unused memory
    gc()

    # Make out-of-sample predictions with the model
    predicted_values <- model |>
      predict((data_bake) |> filter(month >= end_date + years(8) & month < end_date + years(9)) |> select(-month, -ret_excess, -permno, -mktcap_lag) |> as.matrix()) |>
      as.vector() |>
      tibble(pred = _) |>
      bind_cols(data_bake |> filter(month >= end_date + years(8) & month < end_date + years(9))) |> select(pred, month, permno, mktcap_lag, ret_excess)

    # Remove data we do not need
    rm(model)

    # Clear unused memory
    gc()

    # Add the predictions to the average_predictions (sum up predictions)
    if (is.null(average_predictions)) {
      average_predictions <- predicted_values$pred
    } else {
      average_predictions <- average_predictions + predicted_values$pred
    }


  }


  # Clear unused memory
  gc()

  # Calculate the average prediction across the 10 iterations (10 seeds)
  average_predictions <- average_predictions / 10


  # Combine the average predictions with the other relevant variables
  predicted_values <- cbind(data_bake |> filter(month >= end_date + years(8) & month < end_date + years(9)) |> select(month, permno, mktcap_lag, ret_excess), pred = average_predictions)

  # Removing date we do not need
  rm(average_predictions)

  # Save file locally and remove it from memory
  write.csv(predicted_values, paste0("C:/Users/bjark/Desktop/Speciale/R results/MLdata_", i, ".csv"), row.names=FALSE)
  rm(predicted_values)

  #Clear unused memory
  gc()

}

# Timing how long the code takes to run
Endtime <- Sys.time()
Startime
Endtime


