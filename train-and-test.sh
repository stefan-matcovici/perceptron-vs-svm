#!/usr/bin/env bash
declare -a cs=( 0.0000000000001 0.000000000001 0.00000000001 0.0000000001 )
for i in "${cs[@]}"
do
    #./svm_learn.exe -c $i data/p3/train-01-images.svm model_$i
    #./svm_classify.exe data/p3/train-01-images.svm model_$i predictions_train_$i
    #./svm_classify.exe data/p3/test-01-images.svm model_$i predictions_test_$i
    ./svm_learn.exe -c $i data/p3/train-01-images-W.svm modelW_$i
    ./svm_classify.exe data/p3/test-01-images.svm modelW_$i predictionsW_test_$i
    ./svm_classify.exe data/p3/train-01-images.svm modelW_$i predictionsW_train_$i
done