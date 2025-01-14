#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:j:f:s:m:r:i:p:o:a:q:n:x:u:d:b:l:w:z:g:y:c:e: flag
do
    case "${flag}" in

        m) model=${OPTARG};;
        f) dataset=${OPTARG};;
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        o) outfile=${OPTARG};;
        u) nq=${OPTARG};;
        d) qd=${OPTARG};;
        i) iterations=${OPTARG};;
        b) b=${OPTARG};;
        w) wd=${OPTARG};;
        s) seed=${OPTARG};;
        p) optimiser=${OPTARG};;
        l) lr=${OPTARG};;
        z) slr=${OPTARG};;
        g) g=${OPTARG};;
        r) runs=${OPTARG};;

        j) validation=${OPTARG};;
    esac
done

echo "Model name: $model";
echo "Optimiser: $optimiser";

echo "Number of qubits: $nq";

echo "Training dataset: $train";
echo "Test dataset: $test";
echo "Validation dataset: $validation";
echo "Output file path: $outfile";

echo "Number of epochs: $epochs";
echo "Number of runs: $runs";
echo "Number of iterations: $iterations";
echo "Batch size: $b";



if [[ "${model}" == "alpha_3_multiclass" ]]
then
echo "Running ALPHA3-multiclass"
python3.10 ./data/data_processing/use_alpha_3_multiclass.py -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

elif [[ "${model}" == "alpha_3_multiclass_tests" ]]
then
echo "Running ALPHA3-multiclass-tests"
python3.10 ./data/data_processing/use_alpha_3_multiclass_tests.py -s ${seed} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

elif [[ "${model}" == "beta_2" ]]
then
echo "Running BETA2"
python3.10 ./data/data_processing/use_beta_2_3.py -m beta_2 -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

elif [[ "${model}" == "beta_2_tests" ]]
then
echo "Running BETA2-tests"
python3.10 ./data/data_processing/use_beta_2_3_tests.py -m beta_2_tests -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

elif [[ "${model}" == "beta_3" ]]
then
echo "Running BETA3"
python3.10 ./data/data_processing/use_beta_2_3.py -m beta_3 -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

elif [[ "${model}" == "beta_3_tests" ]]
then
echo "Running BETA3-tests"
python3.10 ./data/data_processing/use_beta_2_3_tests.py -m beta_3_tests -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

else
echo "Invalid model choice, try again!";
fi
