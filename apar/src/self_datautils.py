import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataset.dataset import apar_load_pretrain_dataset, vime_load_pretrain_dataset

def load_data(args, task=None):
    config = getattr(sys.modules[__name__], args.dataset)(args)
    
    # for apar pre-train task
    if task is None:
        config["arithm_op"] = args.arithm_op
        return apar_load_pretrain_dataset(**config)
    
    # for vime pre-train task
    config["model_corr_prob"] = args.model_corr_prob
    config["merge_cat_num_feat"] = False
    return vime_load_pretrain_dataset(**config)

def tw_hp_b(args):
    all_county = [
        'KEL','ILA','CYI_S','HSZ_B','ZMI','NTC','CHW',
        'HSZ_S','YUN','CYI_B','PIF','HUN','TTT','KNH','PEH'
    ]
    all_name = [f"{county}_BLD" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['BLD'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_a(args):
    all_county = [
        'KEL','ILA','CYI_S','HSZ_B','ZMI','NTC','CHW',
        'HSZ_S','YUN','CYI_B','PIF','HUN','TTT','KNH','PEH'
    ]
    all_name = [f"{county}_APT" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['APT'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_h(args):
    all_county = [
        'KEL','ILA','CYI_S','HSZ_B','ZMI','NTC','CHW',
        'HSZ_S','YUN','CYI_B','PIF','HUN','TTT','KNH','PEH'
    ]
    all_name = [f"{county}_HOS" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['HOS'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_ntpc_b(args):
    all_county = ['NTPC']
    all_name = [f"{county}_BLD" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['BLD'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_ntpc_a(args):
    all_county = ['NTPC']
    all_name = [f"{county}_APT" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['APT'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_ntpc_h(args):
    all_county = ['NTPC']
    all_name = [f"{county}_HOS" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['HOS'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_txg_b(args):
    all_county = ['TXG']
    all_name = [f"{county}_BLD" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['BLD'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_txg_a(args):
    all_county = ['TXG']
    all_name = [f"{county}_APT" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['APT'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def tw_hp_txg_h(args):
    all_county = ['TXG']
    all_name = [f"{county}_HOS" for county in all_county]
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/tw_hp/parquet",
        ### for tw_hp dataset ##################
        "county_train": all_county,
        "building_type_train": ['HOS'],
        "county_building_type_test": all_name,
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def ca(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/ca/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def bs(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/bs/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def yp(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/yp/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }



def np(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/np/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def kp(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/kp/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def fp(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/fp/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def villa(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/villa/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def sc(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/sc/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def gs(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/gs/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def er(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/er/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }

def pm(args):
    return {
        "dataset": args.dataset,
        "root_path": "../../dataset/pm/parquet",
        ### for tw_hp dataset ##################
        "county_train": [],
        "building_type_train": [],
        "county_building_type_test": [],
        ########################################
        "valid_ratio": args.val_ratio,
        "numerical_feature_encoding": args.numerical_feature_encoding,
        "categorical_feature_encoding": args.categorical_feature_encoding,
        "target_encoding": args.target_encoding,
        "train_bs": args.batch_size,
        "test_bs": 1,
        "max_train_sample": -1
    }