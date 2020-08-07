# COVID-Analysis
Various files and scripts used to analyse COVID-19 data on a daily basis.

These files explore various modelling techniques to assess the general trend of positive confirmed cases of COVID-19 throughout a range of geographical regions in the United Kingdom. It includes modelling using the following statistical methods:

- Polynomial Regression (note - extreme poor fits, only included as initial experimentation)
- Regression using Natural Cubic Spline (Piece-Wise) Regression Models.
- Time-series forecasting using Auto Regressive Integrated Moving Average (ARIMA) models.

For each technique the range of plots for each region are output into distinct excel files for further analysis and visualisation.

### Examples

![example of cubic spline plots](example_images/exported_plots_example.png?raw=True "Example plots - Cubic Spline Plots automatically formed and exported.")

![example of cubic spline plots](example_images/england_sample_cubic_splines.png?raw=True "Example plots - Sample England Cubic Spline plots.")

![example of cubic spline plots](example_images/bedfordshire_cubic_spline.png?raw=True "Example plots - Bedfordshire Cubic Spline.")

![example of ARIMA plots](example_images/glasgow_ARIMA.png?raw=True "Example plots - ARIMA forecast plot.")

![example of output spreadsheet forecasts](example_images/output_spreadsheet_example.png?raw=True "Example plots - Output spreadsheet figures sample.")

Future work that could significantly improve the effectiveness of this work in providing leading indicators could be to include additional features taken from external data sources, such as mobility data throughout each region, along with population densities, average age of population etc. By introducing this into the models and forming multi-dimensional regression models we could obtain higher assurance on an emerging trend in a given area. This crucially relies on any external data used being relevant, accurate and timely to the original data we have used.

**Disclaimer:** These models should not be relied upon in any way for predictions or accurate assessments of COVID-19 cases. They should only be used for personal use and/or interest only. Production of an accurate COVID-19 prediction model would require a range of high-quality, timely and accurate data from many sources, and would require much greater diligence and use of appropriate pandemic models developed in-line with subject matter expertise.
