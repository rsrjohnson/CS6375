CS6375 Term Project readme file.
Derek Carpenter / dxc190017
Randy Suarez Rodes / rxs179030


-python version 3.7

-All the code is on one file. You need the following libraries:

numpy, pandas, matplotlib.pyplot, sklearn

-Please, read the comments and the report for better understanding.

-Data used: UCI Epileptic Seizure Recognition Data Set.  Hosted on: https://raw.githubusercontent.com/rsrjohnson/CS6375/master/data.csv

-Uncomment run_experiments() to generate the experiments presented on the report. The experiments full execution can take several minutes.

-A user can input desired parameters. Default values are suggested. If any unsupported value is entered, the value of that parameter will be set to a default. Also, an option of verbose is provided to show the training and testing loss and accuracy during the learning process.


NOTE: The algorithm execution time depends on the number epochs and extra layers that are used for the LSTM, be aware that for many layers and many neurons per layer, execution time could be high.


--------------------------------------------------------------------------------------------------------------------
