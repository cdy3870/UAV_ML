#!/bin/bash

################################################### SAMPLING ##########################################################

dir="/official_experiments//sampling//"
parse="parsed_data.txt"

##### Sampling 1: Average (No Chunking) #####

#python timestamp_tests.py -p "$parse" -s 1 -ni 50 -X "${dir}X_sampling_1_50_ints.txt"
#python timestamp_tests.py -p "$parse" -s 1 -ni 200 -X "${dir}X_sampling_1_200_ints.txt"
#python timestamp_tests.py -p "$parse" -s 1 -ni 500 -X "${dir}X_sampling_1_500_ints.txt"

output_file="${dir}ex_average_sampling_ints.csv"

#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Average sampling with 50 intervals" -X "${dir}X_sampling_1_50_ints.txt" -Y Y_data_7_multi.txt -csv $output_file
python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Average sampling with 50 intervals" -X "${dir}X_sampling_1_50_ints.txt" -Y Y_data_7_multi.txt
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Average sampling with 200 intervals" -X "${dir}X_sampling_1_200_ints.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Average sampling with 500 intervals" -X "${dir}X_sampling_1_500_ints.txt" -Y Y_data_7_multi.txt -csv $output_file

##### Sampling 2: Uniform (No Chunking) #####
#python timestamp_tests.py -p "$parse" -s 2 -ni 50 -X "${dir}X_sampling_2_50_ints.txt"
#output_file="${dir}ex_direct_sampling_ints.csv"
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Direct sampling with 50 intervals" -X "${dir}X_sampling_2_50_ints.txt" -Y Y_data_7_multi.txt -csv $output_file


##### Sampling 3: Local Window (No Chunking) #####

#python timestamp_tests.py -p "$parse" -s 3 -ni 50 -wd 2 -X "${dir}X_sampling_3_50_ints_2_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 50 -wd 5 -X "${dir}X_sampling_3_50_ints_5_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 50 -wd 10 -X "${dir}X_sampling_3_50_ints_10_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 50 -wd 20 -X "${dir}X_sampling_3_50_ints_20_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 50 -wd 30 -X "${dir}X_sampling_3_50_ints_30_wd.txt"

output_file="${dir}ex_window_sampling_50_ints.csv"

#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 50, 2 second intervals" -X "${dir}X_sampling_3_50_ints_2_wd.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 50, 5 second intervals" -X "${dir}X_sampling_3_50_ints_5_wd.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 50, 10 second intervals" -X "${dir}X_sampling_3_50_ints_10_wd.txt" -Y Y_data_7_multi.txt -csv $output_file 
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 50, 20 second intervals" -X "${dir}X_sampling_3_50_ints_20_wd.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 50, 30 second intervals" -X "${dir}X_sampling_3_50_ints_30_wd.txt" -Y Y_data_7_multi.txt -csv $output_file

#python timestamp_tests.py -p "$parse" -s 3 -ni 200 -wd 2 -X "${dir}X_sampling_3_200_ints_2_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 200 -wd 5 -X "${dir}X_sampling_3_200_ints_5_wd.txt"
#python timestamp_tests.py -p "$parse" -s 3 -ni 200 -wd 10 -X "${dir}X_sampling_3_200_ints_10_wd.txt"

output_file="${dir}ex_window_sampling_200_ints.csv"

#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 200, 2 second intervals" -X "${dir}X_sampling_3_200_ints_2_wd.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 200, 5 second intervals" -X "${dir}X_sampling_3_200_ints_5_wd.txt" -Y Y_data_7_multi.txt -csv $output_file          
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 200, 10 second intervals" -X "${dir}X_sampling_3_200_ints_10_wd.txt" -Y Y_data_7_multi.txt -csv $output_file  

#python timestamp_tests.py -p "$parse" -s 3 -ni 500 -wd 2 -X "${dir}X_sampling_3_500_ints_2_wd.txt"                                                                                                          
#python timestamp_tests.py -p "$parse" -s 3 -ni 500 -wd 5 -X "${dir}X_sampling_3_500_ints_5_wd.txt"                                                                                                          
#python timestamp_tests.py -p "$parse" -s 3 -ni 500 -wd 10 -X "${dir}X_sampling_3_500_ints_10_wd.txt"                                                                                                        

output_file="${dir}ex_window_sampling_500_ints.csv"                                                                                                                                                         

#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 500, 2 second intervals" -X "${dir}X_sampling_3_500_ints_2_wd.txt" -Y Y_data_7_multi.txt -csv $output_file        
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 500, 5 second intervals" -X "${dir}X_sampling_3_500_ints_5_wd.txt" -Y Y_data_7_multi.txt -csv $output_file
#python LSTM_implementation.py -n 7 -l 0.001 -e 100 -sh -k 10 -st -d "Window sampling with 500, 10 second intervals" -X "${dir}X_sampling_3_500_ints_10_wd.txt" -Y Y_data_7_multi.txt -csv $output_file 


############################################ HANDLING IMBALANCES ##################################################
dir="/official_experiments//handling_imbalances//"

output_file="${dir}ex_ros.csv" 
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "1.5x minority classes" -sample ros -r 100 150 150 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2x minority classes" -sample ros -r 100 200 200 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2.5x minority classes" -sample ros -r 100 250 250 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file

output_file="${dir}ex_smote.csv"
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "1.5x minority classes" -sample smote -r 100 150 150 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2x minority classes" -sample smote -r 100 200 200 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2.5x minority classes" -sample smote -r 100 250 250 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file

output_file="${dir}ex_rus.csv"
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.75x majoirty class" -sample rus -r 75 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.50x majoirty class" -sample rus -r 50 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.25x majoirty class" -sample rus -r 25 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file

output_file="${dir}ex_nn.csv"
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.75x majoirty class" -sample nn -r 75 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.50x majoirty class" -sample nn -r 50 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "0.25x majoirty class" -sample nn -r 25 100 100 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file

output_file="${dir}ex_data_aug.csv"
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "1.5x minority classes" -a 150 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2x minority classes" -a 200 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file
#python LSTM_implementation.py -n 7 -e 100 -k 10 -st -d "2.5x minority classes" -a 250 -X "X_data_7_multi.txt" -Y "Y_data_7_multi.txt" -csv $output_file