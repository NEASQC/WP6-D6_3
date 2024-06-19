#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:j:f:s:m:r:i:p:o:a:q:n:x:u:d:b:l:w:z:g:y:c:e: flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        j) validation=${OPTARG};;
        f) dataset=${OPTARG};;
        s) seed=${OPTARG};;
        m) model=${OPTARG};;
        r) runs=${OPTARG};;
        i) iterations=${OPTARG};;
        p) optimiser=${OPTARG};;
        o) outfile=${OPTARG};;
        a) ansatz=${OPTARG};;
        q) qn=${OPTARG};;
        n) nl=${OPTARG};;
        x) np=${OPTARG};;

        u) nq=${OPTARG};;
        d) qd=${OPTARG};;
        b) b=${OPTARG};;
        l) lr=${OPTARG};;
        w) wd=${OPTARG};;
        z) slr=${OPTARG};;
        g) g=${OPTARG};;

        y) version=${OPTARG};;
        c) pca=${OPTARG};;
        e) qs=${OPTARG};;
    esac
done

echo "train: $train";
echo "test: $test";
echo "validation: $validation";
echo "seed: $seed";
echo "model: $model";
echo "epochs: $epochs";
echo "runs: $runs";
echo "optimiser: $optimiser";
echo "iterations: $iterations";
echo "outfile: $outfile";
echo "ansatz: $ansatz";
echo "Number of qubits per noun: $qn";
echo "number of circuit layers: $nl";
echo ":number of single qubit parameters $np";

echo "Number of qubits in our circuit: $nq";
echo "Initial spread of the parameters: $qd";
echo "Batch size: $b";
echo "Learning rate: $lr";
echo "Weight decay: $wd";
echo "Step size for the learning rate scheduler: $slr";
echo "Gamma for the learning rate scheduler: $g";

echo "Version between alpha_1 and alpha_2: $version";
echo "Reduced dimension for the word embeddings: $pca";
echo "Number of qubits per SENTENCE type: $qs";



if [[ "${model}" == "pre_alpha_1" ]]
then
echo "running pre_alpha"
python3.10 ./data/data_processing/use_pre_alpha_1.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -val ${validation} -o ${outfile}
elif [[ "${model}" == "pre_alpha_2" ]]
then
echo "running pre_alpha_2"
python3.10 ./data/data_processing/use_pre_alpha_2.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -val ${validation} -o ${outfile} -an ${ansatz} -qn ${qn} -nl ${nl} -np ${np} -b ${b}
elif [[ "${model}" == "alpha_3" ]]
then
echo "running alpha_3"
python3.10 ./data/data_processing/use_alpha_3.py -s ${seed} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -val ${validation} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "alpha_1_2" ]]
then
echo "running alpha_1_2"
python3.10 ./data/data_processing/use_alpha_1_2.py -s ${seed} -i ${iterations} -r ${runs} -v ${version} -pca ${pca} -tr ${train} -te ${test} -val ${validation} -o ${outfile} -an ${ansatz} -qn ${qn} -qs ${qs} -nl ${nl} -np ${np} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "alpha_3_multiclass" ]]
then
echo "running alpha_3_multiclass"
python3.10 ./data/data_processing/use_alpha_3_multiclass.py -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "alpha_3_multiclass_tests" ]]
then
echo "running alpha_3_multiclass_tests"
python3.10 ./data/data_processing/use_alpha_3_multiclass_tests.py -s ${seed} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "beta_2" ]]
then
echo "running beta_2"
python3.10 ./data/data_processing/use_beta_2_3.py -m beta_2 -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "beta_2_tests" ]]
then
echo "running beta_2_tests"
python3.10 ./data/data_processing/use_beta_2_3_tests.py -m beta_2_tests -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "beta_3" ]]
then
echo "running beta_3"
python3.10 ./data/data_processing/use_beta_2_3.py -m beta_3 -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "beta_3_tests" ]]
then
echo "running beta_3_tests"
python3.10 ./data/data_processing/use_beta_2_3_tests.py -m beta_3_tests -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${dataset} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
else
echo "no model ran";
fi
