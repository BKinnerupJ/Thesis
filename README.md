# Thesis
Machine Learning Portfolios - Margin or Mirage

The four R scripts in this repository creates all the results for our Master Thesis: Machine Learning Portfolio - Margin or Mirage?. 


01 Machine Learning model predictions:  This script trains our ML model and generates monthly stock predictions.
 				                                The data used can be obtained from CRSP and by following: https://voigtstefan.quarto.pub/course-exercises/gu_kelly_xiu.html  


02 Average daily dollar volume:         This script calculates the average daily dollar volume used for transaction cost calculation using daily data from CRSP



03 Risk free rate:		                  This script retrieves the risk free rate used to convert monthly returns into monthly excess return using data from CRSP



04 Backtesting:			                    This script constructs all our backtesting results. It relies on the output of the three previous scripts and on a tidy_finance_ML.sqlite file that can be created with
			                                  data from CRSP and by following: https://voigtstefan.quarto.pub/course-exercises/gu_kelly_xiu.html  


Peter Tøjner Götke and Bjarke Kinnerup Jørgesensen, May 2024. Copenhagen. 
