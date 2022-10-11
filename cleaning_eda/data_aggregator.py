# create stand-alone aggregate script
# 1 output: path to out_file

# join raw files from 2016-2020, keep only common columns


import argparse
import pandas as pd

def aggregate():
    
    keep_columns = ['Control\nNumber', 'Species\nCode', 'Genus', 'Species','Subspecies','Specific\nName', 'Generic\nName', 'Wildlf\nDesc', 'Wildlf\nCat','Cartons', 'Qty', 'Unit', 'Value', 'Ctry\nOrg', 'Ctry\nIE', 'Purp','Src', 'Trans Mode', 'Act', 'Dp\nCd', 'Disp\nDate', 'Ship\nDate','I\nE', 'Pt\nCd']
    
    
    
    
    
    
    
    
    

    print(combined_df.shape)

    return combined_df






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Combined data file (CSV)')
    args = parser.parse_args()

    aggregated = aggregate()
    aggregated.to_csv(args.output, index=False)


#run python respondent_data_clean.py data/respondent_contact.csv data/respondent_other.csv data/respondent_combined.csv