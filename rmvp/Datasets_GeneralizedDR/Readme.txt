
The folder contains the daily prices (adjusted for splits and dividends) for the following markets:

- Dow Jones Industrial from 06/10/2006 to 24/02/2023;
- Euro Stoxx 50 from 06/10/2006 to 24/02/2023;
- FTSE 100 from 06/10/2006 to 24/02/2023;
- NASDAQ 100 from 06/10/2006 to 24/02/2023;
- S&P500 from 06/10/2006 to 24/02/2023.

The data are available both in .xlsx and in .mat extensions.

For each market listed above, the .xlsx file contains:

- the sheet 'AssetPrices', namely a matrix where the first column represents the vector of dates, while the remaining columns are the daily prices of each asset.
- the sheet 'IndexPrices', namely a matrix where the first column represents the vector of dates, while the second column indicates the daily prices of the market index.

The .mat file is a workspace containing:

- the matrix 'AssetPrices', where each column is the daily prices of each asset
- the vector 'IndexPrices', representing the daily prices of the market index
- the cellarray 'TimeP', where are listed the dates
- the cellarray 'Names', where are reported the constituents.
