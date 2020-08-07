# COVID-Analysis
Various files and scripts used to analyse COVID-19 data throghout the United Kingdom on a daily basis.

These files explore various modelling techniques to assess the general trend of positive confirmed cases of COVID-19 throughout a range of geographical regions in the United Kingdom. 

### Data Preprocessing

The data preprocessing script (load_and_preprocess.py) automatically scrapes the newest data for England, Scotland, Ireland, and Wales. Unfortunately, due to the means in which the data is hosted on a dynamic dashboard, the data for Northern Ireland, Jersey, Isle of Man, and Guernsey was is more difficult to obtain and had to be scraped in a more complex manner using selenium. The script for this is not included in this repo, however the scraped data for this locations has been inluded in the file 'Misc_scraped_positive_cases.csv'.

All of the positive cases data obtained for each area throughout the UK is corrected so that it is Cases Per 100,000 population, which is more informative for analysing and comparing cases between sub-regional locations.

### Analysis and Processing of Trends and Forecasts

The subsequent processing and analysis on this data includes use of the following statistical methods: 

- Polynomial Regression (note - extreme poor fits, only included as initial experimentation)
- Regression using Natural Cubic Spline (Piece-Wise) Regression Models.
- Time-series forecasting using Auto Regressive Integrated Moving Average (ARIMA) models.

For each technique the range of plots for each region are output into distinct excel files for further analysis and visualisation. This is conducted for all of the UK locations mentioned above previously.

### Examples

Natural Cubic Spline plots and ARIMA plots are formed for all individual local authority areas within each of the UK areas (England, Scotland, Wales, Northern Ireland, Ireland, Jersey, Isle of Man, and Guernsey). A sample of these are shown below:

![example of cubic spline plots](example_images/exported_plots_example.png?raw=True "Example plots - Cubic Spline Plots automatically formed and exported.")

![example of cubic spline plots](example_images/england_sample_cubic_splines.png?raw=True "Example plots - Sample England Cubic Spline plots.")

![example of cubic spline plots](example_images/bedfordshire_cubic_spline.png?raw=True "Example plots - Bedfordshire Cubic Spline.")

![example of ARIMA plots](example_images/glasgow_ARIMA.png?raw=True "Example plots - ARIMA forecast plot.")

In addition, the positive cases per 100,000 population figures are presented for all locations within the final Excel spreadsheet produced. This includes 5 day forward forecasts and a calculation of the gradient for the final forecasts. This gives an indication towards the relative trend of cases for a given area.

![example of output spreadsheet forecasts](example_images/output_spreadsheet_example.png?raw=True "Example plots - Output spreadsheet figures sample.")

Future work that could significantly improve the effectiveness of this work in providing leading indicators could be to include additional features taken from external data sources, such as mobility data throughout each region, along with population densities, average age of population etc. By introducing this into the models and forming multi-dimensional regression models we could obtain higher assurance on an emerging trend in a given area. This crucially relies on any external data used being relevant, accurate and timely to the original data we have used.

In addition, a dashboard could be used to more effectively present this data. The above work was relatively straightforward to implement as a local dashboard using Dash. 

**Disclaimer:** These models should not be relied upon in any way for predictions or accurate assessments of COVID-19 cases. They should only be used for personal use and/or interest only. Production of an accurate COVID-19 prediction model would require a range of high-quality, timely and accurate data from many sources, and would require much greater diligence and use of appropriate pandemic models developed in-line with subject matter expertise.
