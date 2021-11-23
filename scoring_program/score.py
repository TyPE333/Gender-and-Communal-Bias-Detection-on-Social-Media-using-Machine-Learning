    #!/usr/bin/env python

import os
import sys
from sys import argv

import libscores
import yaml

import pandas as pd
from sklearn.metrics import f1_score
import sklearn

from libscores import ls, filesep, mkdir, read_array, compute_all_scores, write_scores

# Default I/O directories:
root_dir = "/home/ritesh/Dropbox/kmi-event-related-stuffs/ICON2021/ST"
default_input_dir = os.path.join(root_dir, "data")
default_output_dir = os.path.join(root_dir, "scores")

# Constant used for a missing score
missing_score = -0.999999

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 0

def validate_format(gold, prediction):
    pred_cols = prediction.columns
    gold_cols = gold.columns

    print ('Col length', len(pred_cols), len(gold_cols))
    
    #Checking if headers are same or not
    if (list(pred_cols) != list(gold_cols)): raise ValueError(
        "Bad prediction column header. Prediction header: {}\nExpected Headers:{}".format(pred_cols, gold_cols))

    #Checking for number of columns
    if (len(gold_cols) != len(pred_cols)): raise ValueError(
        "Unexpected number of columns. Prediction Column Count: {}\nExpected Column Count:{}".format(str(len(pred_cols)), str(len(gold_cols))))

    #Checking for number of rows
    if (len(gold) != len(prediction)): raise ValueError(
        "Unexpected number of prediction instances. Prediction Instances Count: {}\nExpected Column Count:{}".format(str(len(prediction)), str(len(gold))))


def get_microf1(df_merged):
    try:
        #Compute the Micro F1 Score

        #Separating label of each level
        df_merged['agg_pred'] = df_merged['Labels_y'].apply(lambda x: x.split(',')[0].replace('(','').strip())
        df_merged['gen_pred'] = df_merged['Labels_y'].apply(lambda x: x.split(',')[1].strip())
        df_merged['com_pred'] = df_merged['Labels_y'].apply(lambda x: x.split(',')[2].replace(')','').strip())

        df_merged['agg_gold'] = df_merged['Labels_x'].apply(lambda x: x.split(',')[0].replace('(','').strip())
        df_merged['gen_gold'] = df_merged['Labels_x'].apply(lambda x: x.split(',')[1].strip())
        df_merged['com_gold'] = df_merged['Labels_x'].apply(lambda x: x.split(',')[2].replace(')','').strip())

        print (df_merged.head())
        #Compute Micro F1 for Aggression
        agg_true = list(df_merged['agg_gold'])
        agg_pred = list(df_merged['agg_pred'])
        micro_f1_agg = f1_score(agg_true, agg_pred, average='micro')
        print(
            "======= Aggression Score(Micro F1)=%0.12f =======" % micro_f1_agg)

        #Compute Micro F1 for Gender Bias
        gen_true = list(df_merged['gen_gold'])
        gen_pred = list(df_merged['gen_pred'])
        micro_f1_gen = f1_score(gen_true, gen_pred, average='micro')
        print(
            "======= Gender Bias Score(Micro F1)=%0.12f =======" % micro_f1_gen)  

        #Compute Micro F1 for Communal Bias
        com_true = list(df_merged['com_gold'])
        com_pred = list(df_merged['com_pred'])
        micro_f1_com = f1_score(gen_true, gen_pred, average='micro')
        print(
            "======= Communal Bias Score(Micro F1)=%0.12f =======" % micro_f1_agg) 

        #Compute Overall Micro f1
        overall_micro_f1 = (micro_f1_agg + micro_f1_gen + micro_f1_com) / 3
        print(
            "======= Overall Score(Micro F1)=%0.12f =======" % overall_micro_f1)
    except Exception as e:
        print (e)
        raise Exception('Error in calculation of the Micro F1 of the task')
        
    return overall_micro_f1, micro_f1_agg, micro_f1_gen, micro_f1_com

def get_instancef1(df_merged):
    try:
        # Compute the Instance F1 Score            
        y_true = list(df_merged['Labels_x'])
        y_pred = list(df_merged['Labels_y'])
        instance_f1 = f1_score(y_true, y_pred, average='micro')
        print('Instance F1: ', instance_f1)

        print(
            "======= Score(instance)=%0.12f =======" % instance_f1)
    
    except:
        raise Exception('Error in calculation of the Instance F1 of the task')

    return instance_f1

if __name__ == "__main__":

    print('The Python version is {}.'.format(sys.version))
    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        print ('Defaulting to default dirs')
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        # Create the output directory, if it does not already exist and open output files
    print ('Input directory', input_dir)
    print ('Output directory', output_dir)

    mkdir(output_dir)
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(input_dir, 'ref', '*.tsv')))
    prediction_names = sorted(ls(os.path.join(input_dir, 'res', '*.tsv')))
    # solution_name = os.path.join(output_dir, 'gold_labels.tsv')
    print ('Gold files', solution_names)
    for solution_name in solution_names:
        try:
            # predict_file = ls(os.path.join(input_dir, 'res', '*.tsv'))[-1]
            predict_file = solution_name.replace('dev', 'pred').replace('ref', 'res')
            print ('Pred file', predict_file)
            # if (predict_file == []): raise IOError('Missing prediction file')
            if os.path.exists(predict_file):
                print ('Pred file exists', predict_file)
                lang_name = predict_file[predict_file.rfind('_')+1:predict_file.rfind('.tsv')]
                print ('Lang name', lang_name)

                gold = pd.read_csv(solution_name, delimiter='\t')
                prediction = pd.read_csv(predict_file, delimiter='\t')

                validate_format(gold, prediction)           

                df_merged = pd.merge(gold, prediction, left_on='ID', right_on='ID')
                print (df_merged.head())

                #Calculating and writing micro f1
                overall_micro_f1, micro_f1_agg, micro_f1_gen, micro_f1_com = get_microf1(df_merged)

                html_file.write(
                "======= %s Aggression Micro F1 %0.12f: =======\n" % (lang_name, micro_f1_agg))

                html_file.write(
                "======= %s Gender Bias Micro F1 %0.12f: =======\n" % (lang_name, micro_f1_gen))

                html_file.write(
                "======= %s Communal Bias Micro F1 %0.12f: =======\n" % (lang_name, micro_f1_com))
        
                html_file.write(
                "======= %s Overall Micro F1 %0.12f: =======\n" % (lang_name, overall_micro_f1))

                score_file.write("%s_agg_microf1: %0.12f\n" % (lang_name, micro_f1_agg))
                score_file.write("%s_gen_microf1: %0.12f\n" % (lang_name, micro_f1_gen))
                score_file.write("%s_com_microf1: %0.12f\n" % (lang_name, micro_f1_com))
                score_file.write("%s_microf1: %0.12f\n" % (lang_name, overall_micro_f1))
                
                #Calculating and writing instance f1
                instance_f1 = get_instancef1(df_merged)
                html_file.write(
                "======= %s Instance F1 %0.12f: =======\n" % (lang_name, instance_f1))

                score_file.write("%s_instf1: %0.12f\n" % (lang_name, instance_f1))
            else:
                print ('Pred file do not exist', predict_file)
                print ('All files in prediction', prediction_names)
            
        except Exception as inst:
            score = missing_score
            print(
                "======= Score(instance)=ERROR =======")
            html_file.write(
                "======= Score(instance)=ERROR =======\n")
            print(inst)

        # Read the execution time and add it to the scores:
        # try:
        #     metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        #     score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])
        # except:
        #     score_file.write("Duration: 0\n")
    html_file.close()
    score_file.close()
    
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'r')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'r')
    print ('Final Score and HTML dir', output_dir)
    print ('Final score content', score_file.read())
    print ('Final HTML content', html_file.read())
    html_file.close()
    score_file.close()
        # if debug_mode > 0:
        #     scores = compute_all_scores(solution, prediction)
        #     write_scores(html_file, scores)
