import sklearn.metrics as skm 
import json 
import numpy as np 

def compute_metrics_logs(
        results_path : str,
        kwargs : dict ={
            'precision' : {}, 'accuracy' : {},
            'recall' : {}, 'roc_auc' : {'multi_class' : 'ovr'},
            'confusion_matrix' : {},
            'f1' : {}}):
    """
    For an experiment result, it computes the values of precision, accuracy
    , recall, roc_auc, confusion matrix and f1 score.

    Parameters
    ----------
    results_path : str
        Path with the experiment results
    kwargs : dict
        Optional arguments passed to the scikit-learn functions
        for computing the metrics. 
        It is important to note that some arguments will have to be changed
        for binary/multilabel classification. More info can be found in
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """    

    with open(results_path, 'r') as file:
        r = json.load(file)

    ### Define metric dictionary with metrics functions 
    metric_functions = {
        'precision': skm.precision_score,
        'accuracy' : skm.accuracy_score,
        'recall' : skm.recall_score,
        'roc_auc': skm.roc_auc_score,
        'confusion_matrix': skm.confusion_matrix,
        'f1': skm.f1_score}

    ### Alpha 3// PreAlpha2###
    if r['model_name'] != 'beta_1':

        metrics_results = {}
        runs = r['input_args']['runs']
        iterations = r['input_args']['iterations']
        ### Check if results are binary of multiclass
        labels_keys = ['train_labels', 'val_labels', 'test_labels']
        all_labels = [
            label for key,values in r.items(
            ) if key in labels_keys for label in values
        ]
        n_labels = len(set(all_labels))

        for m in metric_functions.keys():
            metrics_results[f'{m}_train'] = [[] for _ in range(runs)]
            metrics_results[f'{m}_val'] = [[] for _ in range(runs)]
            metrics_results[f'{m}_test'] = []
            for i in range(runs):
                for j in range(iterations):
                    for dataset in ('train', 'val'):
                                
                        if m == 'roc_auc_score':
                            ### AUC score has different behaviour for binary and multiclass
                            if n_labels == 2:
                                metrics_results[f'{m}_{dataset}'][i].append(
                                metric_functions[m](
                                r[f'{dataset}_labels'],
                                np.array(r[
                                    f'{dataset}_probabilities'][i][j])[:,1],
                                **kwargs[m]))
                                if j == 0:
                                    metrics_results[f'{m}_test'].append(
                                        metric_functions[m](
                                        r['test_labels'],
                                        np.array(r[
                                            'test_probabilities'][i])[:,1],
                                        **kwargs[m]))
                            else:
                                metrics_results[f'{m}_{dataset}'][i].append(
                                metric_functions[m](
                                r[f'{dataset}_labels'],
                                np.array(r[f'{dataset}_probabilities'][i][j]),
                                **kwargs[m]))
                                if j == 0:
                                    metrics_results[f'{m}_test'].append(
                                        metric_functions[m](
                                        r['test_labels'],
                                        np.array(r['test_probabilities'][i]),
                                        **kwargs[m]))
                                    
                        else:
                            metrics_results[f'{m}_{dataset}'][i].append(
                                metric_functions[m](
                                r[f'{dataset}_labels'],
                                r[f'{dataset}_predictions'][i][j],
                                **kwargs[m])
                            )
                            if j == 0:
                                metrics_results[f'{m}_test'].append(
                                    metric_functions[m](
                                    r['test_labels'],
                                    r['test_predictions'][i],
                                    **kwargs[m]))
                        
    ### Beta 1  ####
    if r['model_name'] == "beta_1":
        metric_results = {}
        for m in metric_functions.keys():
                                
            if m == 'roc_auc_score':
                if n_labels == 2:
                    metrics_results[f'{m}_test'].append(
                    metric_functions[m](
                    r['test_labels'],
                    np.array(r['test_probabilities'][i])[:,1],
                    **kwargs[m]))
                else:
                    metrics_results[f'{m}_test'].append(
                    metric_functions[m](
                    r['test_labels'],
                    np.array(r['test_probabilities'][i]),
                    **kwargs[m]))

            else:
                metrics_results[f'{m}_test'].append(
                    metric_functions[m](
                    r['test_labels'],
                    r['test_predictions'][i],
                    **kwargs[m]))

    
    return metrics_results





