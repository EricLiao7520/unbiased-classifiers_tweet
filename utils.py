import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, random_split, Dataset
from explainer import Explainer
from models import ModelWrapper

import os 
import torch
# from fairness_analysis import TripleLinearClassifier, SingleLinearClassifier, config_loss, config_optimizer, train
# from explainer import Explainer

DATA_PATH = "compas-scores.csv"


def plot_bar(x, height, pos_neg, labels, title, save_name):
    ax = plt.subplot()
    ax.set_ylabel('Importance count')
    colours = np.array(['blue'] * len(x))
    colours[pos_neg > 0] = 'red'
    ax.bar(x, height, color=colours)
    legend_labels = ["Positive attribution" ,"Negative attribution"]
    handles = [plt.Rectangle((0,0),1,1, color="red"), plt.Rectangle((0,0),1,1, color="blue")]
    ax.legend(handles, legend_labels)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(title)
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    
def get_base_dfs():
    date_cols = ["compas_screening_date","dob",
             "c_jail_in","c_jail_out","c_offense_date",
             "v_screening_date","screening_date",
             "vr_offense_date","r_jail_out","r_jail_in",
             "r_offense_date","c_arrest_date"]
    df = pd.read_csv(DATA_PATH, parse_dates=date_cols)

    column_filter = ["id", "juv_fel_count", "compas_screening_date", "c_offense_date",
                    "sex", "age", "age_cat", "race", "c_charge_degree",
                    "c_charge_desc", "days_b_screening_arrest",
                    "decile_score","is_recid","r_offense_date",
                    "c_case_number", "v_decile_score", "is_violent_recid",
                    "vr_offense_date", "score_text", "juv_misd_count", "juv_other_count",
                    "priors_count"]

    df_final = df.loc[:, column_filter].copy()
    df_final = df_final[df_final["c_case_number"] != "NaN"]
    df_final = df_final.loc[(df_final["days_b_screening_arrest"] <30) & (df_final["days_b_screening_arrest"] > -30)]
    df_final = df_final.loc[(df_final["is_recid"]!=-1) & (df["decile_score"]!=-1) & (df_final["v_decile_score"] !=-1)]


    df_final["african-american"] = np.where(
            df_final["race"] == "African-American",1,0
        )
    df_final["caucasian"] = np.where(
            df_final["race"] == "Caucasian",1,0
        )
    df_final["hispanic"] = np.where(
            df_final["race"] == "Hispanic",1,0
        )
    df_final["other"] = np.where(
            df_final["race"] == "Other",1,0
        )
    df_final["asian"] = np.where(
            df_final["race"] == "Asian",1,0
        )
    df_final["native-american"] = np.where(
            df_final["race"] == "Native American",1,0
        )

    race_type = pd.CategoricalDtype(categories=['African-American','Caucasian','Hispanic',
                                                "Other",'Asian','Native American'],ordered=True)
    df_final["race"] = df_final["race"].astype(race_type)

    score_type = pd.CategoricalDtype(categories=["Low","Medium","High"],ordered=True)
    df_final["score_text"] = df_final["score_text"].astype(score_type)

    age_type = pd.CategoricalDtype(categories=["Less than 25","25 - 45","Greater than 45"],ordered=True)
    df_final["age_cat"] = df_final["age_cat"].astype(age_type)

    df_final["less25"] = np.where(
        df_final["age_cat"] == "Less than 25", 1, 0
    )
    df_final["25to45"] = np.where(
        df_final["age_cat"] == "25 - 45", 1, 0
    )
    df_final["greater45"] = np.where(
        df_final["age_cat"] == "Greater than 45", 1, 0
    )

    df_final["male"] = np.where(
        df_final["sex"] == "Male", 1, 0
    )
    df_final["female"] = np.where(
        df_final["sex"] == "Female", 1, 0
    )

    df_final["misdemeanor"] = np.where(
        df_final["c_charge_degree"] == "M", 1, 0
    )
    df_final["felony"] = np.where(
        df_final["c_charge_degree"] == "F", 1, 0
    )

    for col in ["sex","c_charge_degree"]:
        df_final[col] = df_final[col].astype("category")

    # exclude traffic tickets & municipal ordinance violations
    df_final = df_final[df_final["c_charge_degree"] != "O"]
    df_final = df_final[df_final["score_text"] != "NaN"]

    df_final = df_final[df_final["c_offense_date"] < df_final["compas_screening_date"]]

    # check if person reoffended within 2 years
    def two_years(col,col_recid):
        # first we subtract the time columns
        df_final["days"] = df_final[col] - df_final["compas_screening_date"]
        # as it returns a time delta we convert it to int with .days
        df_final["days"] = df_final["days"].apply(lambda x:x.days)
        
        # then we assign the values 0,1 and 3 with np.where ( two years are 730 days )
        df_final["two"] = np.where(df_final[col_recid]==0,0,
                    np.where((df_final[col_recid]==1) & (df_final["days"] < 730),1,1))
        
        return df_final["two"]
        
    # check if person recided
    df_final["two_years_r"] = two_years("r_offense_date","is_recid")
    # check if person recided violentley
    df_final["two_years_v"] = two_years("vr_offense_date","is_violent_recid")

    df_final_c = df_final[df_final["two_years_r"] !=3].copy()
    df_final_v = df_final[df_final["two_years_v"] != 3].copy()

    # binarise decile scores
    df_final_c["binary_decile_score"] = np.where(df_final_c["decile_score"] >=5,1,0)
    df_final_v["binary_v_decile_score"] = np.where(df_final_v["v_decile_score"] >=5,1,0)

    df_final_c.reset_index(drop=True,inplace=True)
    df_final_v.reset_index(drop=True,inplace=True)
    
    return df_final_c, df_final_v
    

class COMPASDataset(Dataset):
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        
        if target_column in df.columns:
            self.feature_columns = df.drop(columns=[target_column])
            self.features = torch.tensor(self.feature_columns.values, dtype=torch.float32)
            self.targets = torch.tensor(df[target_column].values, dtype=torch.float32)
        else:
            self.feature_columns = df
            self.features = torch.tensor(df.values, dtype=torch.float32)
            self.targets = None
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]
    
    def get_columns(self):
        return self.feature_columns.columns.tolist()

def create_data_splits(dataset, train_ratio=0.40, diverse_ratio=0.40, val_ratio=0.0, test_ratio=0.20, batch_size=64, seed=42):
    assert np.isclose(train_ratio + val_ratio + test_ratio + diverse_ratio, 1.0)
    torch.manual_seed(seed)
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    diverse_size = int(diverse_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - diverse_size - train_size - val_size
    train_dataset, diverse_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, diverse_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    div_batch_size = len(diverse_dataset)//len(train_loader) + 1
    diverse_loader = DataLoader(diverse_dataset, batch_size=div_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, diverse_loader, val_loader, test_loader, train_dataset, diverse_dataset, val_dataset, test_dataset

def explain_model(model, num_classes, dataset, model_name, input_dim, train_columns):
    feature_count_all_lime = []
    pos_neg_count_all_lime = []
    feature_count_all_shap = []
    pos_neg_count_all_shap = []
    for datapoint in dataset:
        X_test = datapoint[0].unsqueeze(0)
        X_label = datapoint[1].unsqueeze(0)
        for current_class in range(num_classes):
            class_wrap = ModelWrapper(model, current_class)
            explainer = Explainer(class_wrap)
            feature_count, pos_neg_count = explainer.count_lime(X_test,
                                                        4,
                                                        input_dim, X_label)
            
            if len(feature_count_all_lime) > current_class:
                feature_count_all_lime[current_class] += feature_count
                pos_neg_count_all_lime[current_class] += pos_neg_count
            else:
                feature_count_all_lime.append(feature_count)
                pos_neg_count_all_lime.append(pos_neg_count)
            
            feature_count, pos_neg_count = explainer.count_shap(X_test,
                                                        4,
                                                        input_dim, X_label)
            if len(feature_count_all_shap) > num_classes:
                feature_count_all_shap[current_class] += feature_count
                pos_neg_count_all_shap[current_class] += pos_neg_count
            else:
                feature_count_all_shap.append(feature_count)
                pos_neg_count_all_shap.append(pos_neg_count)
        
        
        
    for current_class in range(num_classes):
        plot_bar(x=np.arange(input_dim),
            height=feature_count_all_lime[current_class],
            pos_neg=pos_neg_count_all_lime[current_class],
            labels=train_columns,
            title='Lime - {} model class {} over {} samples.'.format(model_name, current_class, len(dataset)),
            save_name='lime_{}_{}.png'.format(model_name,current_class))
        plot_bar(x=np.arange(input_dim),
            height=feature_count_all_shap[current_class],
            pos_neg=pos_neg_count_all_shap[current_class],
            labels=train_columns,
            title='Shapely - {} model class {} over {} samples.'.format(model_name, current_class, len(dataset)),
            save_name='shap_{}_{}.png'.format(model_name, current_class))
