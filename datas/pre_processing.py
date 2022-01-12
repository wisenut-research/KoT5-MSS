
import glob
import json
import pandas as pd
import numpy as np
import sys
import argparse
import os


def parse_file(file_path) :
    print("file_path: ", file_path)
    file_names = glob.glob(file_path + "/*")
    file_list_json = [file for file in file_names if file.endswith(".json")]
    print("file_size : {}".format(len(file_list_json)))

    if "c_event" in file_path: domain = 'c_event'
    elif "culture" in file_path: domain = 'culture'
    elif "enter" in file_path: domain = 'enter'
    elif "fm_drama" in file_path: domain = 'fm_drama'
    elif "fs_drama" in file_path: domain = 'fs_drama'
    elif "history" in file_path: domain = 'history'
    else : return 0

    dialogue = []
    summary1 = []
    summary2 = []
    summary3 = []
    domain_list = []
    for file_json in file_list_json :
        domain_list.append(domain)
        json_data = json.load(open(file_json, 'r', encoding='utf-8'))
        tmp_sent = json_data['Meta']['passage']
        tmp_sent = tmp_sent.replace("\t", "")

        dialogue.append(tmp_sent.replace("\n", " "))

        tmp_smr1 = json_data['Annotation']['Summary1'].replace("\t", "")
        tmp_smr2 = json_data['Annotation']['Summary2'].replace("\t", "")
        tmp_smr3 = json_data['Annotation']['Summary3'].replace("\t", "")

        if tmp_smr1 == "" : summary1.append('None')
        else : summary1.append(tmp_smr1)

        if tmp_smr2 == "" : summary2.append('None')
        else : summary2.append(tmp_smr2)

        if tmp_smr3 == "": summary3.append('None')
        else : summary3.append(tmp_smr3)

    df = pd.DataFrame({"domain": domain_list, "context": dialogue, "summary1": summary1, "summary2": summary2, "summary3": summary3})

    return df




def parse_by_task(df, output_length="all", domain="all"):
    domain_dict = {'c_event': '기타, ',
                   'culture': '문화, ',
                   'enter': '예능, ',
                   'fm_drama': '드라마1, ',
                   'fs_drama': '드라마2, ',
                   'history': '역사, '}

    output_length_dict = {'single': '단문장, ',
                   '3sent': '세문장, ',
                   '20per': '이할문장, '}

    if domain not in list(domain_dict.keys()) and domain != 'all' : raise Exception('Check Domain')
    if output_length not in list(output_length_dict.keys()) and output_length != 'all' : raise Exception('Check Output_length')

    def by_output_length(df, output_length) :
        df_output = df.copy()
        df_output['tmp'] = df_output['domain']
        df_output['tmp'] = df_output.replace({"tmp": domain_dict})
        prefix = output_length_dict[output_length] + '요약 : '
        df_output.reset_index(inplace=True)
        df_output = df_output.drop(['summary2', 'summary3'], axis=1)
        df_output.rename(columns={'summary1': 'summary'}, inplace=True)
        df_output = df_output[df_output.summary != "None"]
        df_output['context'] = df_output.replace({"tmp": domain_dict})['tmp'].astype(str) + prefix + + df_output[
            'context'].astype(str)
        df_output.insert(1, 'output_length', output_length)
        df_output = df_output.drop(['tmp'], axis=1)

        return df_output

    df_single = by_output_length(df, 'single')
    df_3sent = by_output_length(df, '3sent')
    df_20per = by_output_length(df, '20per')

    df = pd.concat([df_single, df_3sent, df_20per], axis=0)
    df = df.sort_index()
    df = df.drop(['index'], axis=1)

    if domain != 'all' :
        df = df[df.domain == domain]

    if output_length != 'all' :
        df = df[df.output_length == output_length]

    return df



def main(data_dir, domain, output_length) :
    train_val_test = ['Training', 'Validation', 'Test']

    def parse_dir(folder_list_lv1) :
        df_list = []
        for folder_lv1 in folder_list_lv1:  # domain
            folder_list_lv2 = glob.glob(folder_lv1 + "/*")
            for folder_lv2 in folder_list_lv2:  # ouput_length
                result_df = parse_file(folder_lv2)
                df_list.append(result_df)
        result_df = pd.concat(df_list, axis=0)
        return result_df

    for mode in train_val_test :
        if os.path.isfile('{}/{}.tsv'.format(data_dir, mode)):
            result_df = pd.read_csv('{}/{}.tsv'.format(data_dir, mode), sep='\t')

        else :
            folder_list = glob.glob("{}/{}/*".format(data_dir, mode))
            result_df = parse_dir(folder_list)
            result_df.to_csv('{}/{}.tsv'.format(data_dir, mode), sep='\t',index=False)

        result_df = parse_by_task(result_df, output_length=output_length, domain=domain)
        result_df.to_csv('{}/{}_{}_{}.tsv'.format(data_dir, mode, domain, output_length), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./datas', help='./datas(default)')
    parser.add_argument("--domain", default='all', help='all(default), c_event, culture, enter, fm_drama, fs_drama, history')
    parser.add_argument("--output_length", default='all', help='all(default), single, 20per, 3sent')
    args = parser.parse_args()

    data_dir = args.data_dir
    domain = args.domain
    output_length = args.output_length

    main(data_dir, domain, output_length)