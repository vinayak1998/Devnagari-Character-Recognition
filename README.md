# Devnagari-Character-Recognition
Using CNN(Keras) and Image Augmentation techniques to classify a given set of handwritten Devnagari characters

## Name
cnn - Run the executable program for CNN

## Synopsis
`./cnn <tr> <ts> <out>`

## Description
This program will train CNN model using given code on train data, make predictions on test data
and write final predictions in given output file.

## Options
- **TR**

  File containing training data in csv format where 1st entry is the target
- **TS**

  File containing test data in csv format where 1st entry is the target
- **OUT** 

  Output file for predictions. One value in each line.

## Example
`./cnn train.csv test.csv output`

## Data
- cnn_train.csv: Train data
- cnn_test.csv: Test data