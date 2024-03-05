import numpy as np
import os
import pandas as pd
from ml_collections import config_dict
from experiment_config import default_cfg
import yaml
import pandas as pd

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras

import tensorflow as tf
import tensorflow_datasets as tfds

from wake_vision_loader import get_distance_eval, get_wake_vision, get_lighting, get_miaps, get_hands_feet_eval, get_depiction_eval
from vww_loader import get_vww

def calc_macs(model):
    total_macs = 0
    #calculate the aproximate number of multiply-accumulate operations
    for layer in model.layers:
        type = layer.__class__.__name__
        if type == 'DepthwiseConv2D':
            in_shape = layer.input.shape
            filter_shape = layer.get_config()["kernel_size"]
            out_shape = layer.output.shape
            macs = out_shape[1] * out_shape[2]* out_shape[3] * filter_shape[0] * filter_shape[1]
            total_macs += macs
        elif type == 'Conv2D':
            in_shape = layer.input.shape
            filter_shape = layer.get_config()["kernel_size"]
            out_shape = layer.output.shape
            macs = out_shape[1] * out_shape[2] * out_shape[3] * in_shape[3] * filter_shape[0] * filter_shape[1]
            total_macs += macs
        elif type == 'Dense':
            in_shape = layer.input.shape
            out_shape = layer.output.shape 
            macs = in_shape[1] * out_shape[1]
            total_macs += macs                 
    return total_macs

def get_macs(model):
    result = pd.DataFrame({'macs': [calc_macs(model)]})
    print(result)
    return result
    

def f1(tp_rate, fp_rate, fn_rate):
    return 2 * tp_rate / (2 * tp_rate + fp_rate + fn_rate)

def lighting_eval(model, model_cfg):
    lighting_ds = get_lighting(model_cfg, batch_size=1)

    person_dim_score = model.evaluate(lighting_ds["person_dim"], verbose=0)
    person_normal_light_score = model.evaluate(lighting_ds["person_normal_light"], verbose=0)
    person_bright_score = model.evaluate(lighting_ds["person_bright"], verbose=0)
    
    non_person_dim_score = model.evaluate(lighting_ds["non_person_dim"], verbose=0)
    non_person_normal_light_score = model.evaluate(lighting_ds["non_person_normal_light"], verbose=0)
    non_person_bright_score = model.evaluate(lighting_ds["non_person_bright"], verbose=0)
    
    dim_f1 = f1(person_dim_score[1], 1-non_person_dim_score[1], 1-person_dim_score[1])
    normal_light_f1 = f1(person_normal_light_score[1], 1-non_person_normal_light_score[1], 1-person_normal_light_score[1])
    bright_f1 = f1(person_bright_score[1], 1-non_person_bright_score[1], 1-person_bright_score[1])
    
    result = pd.DataFrame({
        'lighting-dark': [dim_f1],
        'lighting-normal_light': [normal_light_f1],
        'lighting-bright': [bright_f1]})

    print(result)

    return result

def distance_eval(model, model_cfg):
    dist_cfg = model_cfg.copy_and_resolve_references()
    dist_cfg.MIN_BBOX_SIZE = 0.05 #ensure we always use the same min bbox size

    distance_ds = get_distance_eval(dist_cfg)

    near_score = model.evaluate(distance_ds["near"], verbose=0)
    mid_score = model.evaluate(distance_ds["mid"], verbose=0)
    far_score = model.evaluate(distance_ds["far"], verbose=0)
    no_person_score = model.evaluate(distance_ds["no_person"], verbose=0)
    
    near_f1 = f1(near_score[1], 1-no_person_score[1], 1-far_score[1])
    mid_f1 = f1(mid_score[1], 1-no_person_score[1], 1-far_score[1])
    far_f1 = f1(far_score[1], 1-no_person_score[1], 1-far_score[1])

    result = pd.DataFrame({
        'distance-near': [near_f1],
        'distance-mid': [mid_f1],
        'distance-far': [far_f1]})
    print(result)

    return result

def miap_eval(model, model_cfg):
    miap_ds = get_miaps(model_cfg, batch_size=1)

    female_score = model.evaluate(miap_ds["female"], verbose=0)
    male_score = model.evaluate(miap_ds["male"], verbose=0)
    gender_unknown_score = model.evaluate(miap_ds["gender_unknown"], verbose=0)
    young_score = model.evaluate(miap_ds["young"], verbose=0)
    middle_score = model.evaluate(miap_ds["middle"], verbose=0)
    old_score = model.evaluate(miap_ds["older"], verbose=0)
    age_unknown_score = model.evaluate(miap_ds["age_unknown"], verbose=0)
    no_person_score = model.evaluate(miap_ds["no_person"], verbose=0)
    
    female_f1 = f1(female_score[1], 1-no_person_score[1], 1-female_score[1])
    male_f1 = f1(male_score[1], 1-no_person_score[1], 1-male_score[1])
    gender_unknown_f1 = f1(gender_unknown_score[1], 1-no_person_score[1], 1-gender_unknown_score[1])
    
    young_f1 = f1(young_score[1], 1-no_person_score[1], 1-young_score[1])
    middle_f1 = f1(middle_score[1], 1-no_person_score[1], 1-middle_score[1])
    old_f1 = f1(old_score[1], 1-no_person_score[1], 1-old_score[1])
    age_unknown_f1 = f1(age_unknown_score[1], 1-no_person_score[1], 1-age_unknown_score[1])
    
    result = pd.DataFrame({
        'miap-female': [female_f1],
        'miap-male': [male_f1],
        'miap-unknown-gender': [gender_unknown_f1],
        'miap-young': [young_f1],
        'miap-middle': [middle_f1],
        'miap-old': [old_f1],
        'miap-unknown-age': [age_unknown_f1],
        })
    
    print(result)

    return result

def hands_feet_eval(model, model_cfg):
    hands_feet_ds = get_hands_feet_eval(model_cfg, batch_size=1)

    hands_score = model.evaluate(hands_feet_ds["Human hand"], verbose=0)
    feet_score = model.evaluate(hands_feet_ds["Human foot"], verbose=0)
    no_person_score = model.evaluate(hands_feet_ds["no_person"], verbose=0)
    
    hands_f1 = f1(hands_score[1], 1-no_person_score[1], 1-hands_score[1])
    feet_f1 = f1(feet_score[1], 1-no_person_score[1], 1-feet_score[1])
    
    
    result = pd.DataFrame({
        'hands': [hands_f1],
        'feet': [feet_f1],
        })
    
    print(result)

    return result

def depiction_eval(model, model_cfg):
    depiction_ds = get_depiction_eval(model_cfg, batch_size=1)

    person_score = model.evaluate(depiction_ds["person"], verbose=0)
    depictions_persons_score = model.evaluate(depiction_ds["depictions_persons"], verbose=0)
    depictions_non_persons_score = model.evaluate(depiction_ds["depictions_non_persons"], verbose=0)
    non_person_no_depictions_score = model.evaluate(depiction_ds["non_person_no_depictions"], verbose=0)
    
    depictions_persons_f1 = f1(depictions_persons_score[1], 1.-person_score[1], 1-depictions_persons_score[1])
    depictions_non_persons_f1 = f1(depictions_non_persons_score[1], 1-person_score[1], 1-depictions_non_persons_score[1])
    non_person_no_depictions_f1 = f1(non_person_no_depictions_score[1], 1-person_score[1], 1-non_person_no_depictions_score[1])

    result = pd.DataFrame({
        'depictions_persons': [depictions_persons_f1],
        'depictions_non_persons': [depictions_non_persons_f1],
        'non_person_no_depictions': [non_person_no_depictions_f1],
        })
    
    print(result)

    return result


def benchmark_suite(model_cfg, evals=["wv", "vww", "distance", "miap", "lighting", "hands_feet", "depiction", "macs"]):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)
    
    result = pd.DataFrame({'model_name': [model_cfg.MODEL_NAME]})

    if "wv" in evals:
        _, _, wv_test = get_wake_vision(model_cfg)
        wv_test_score = model.evaluate(wv_test, verbose=0)
        print(f"Wake Vision Test Score: {wv_test_score[1]}")
        result = pd.concat([result, pd.DataFrame({"wv_test_score": [wv_test_score[1]]})], axis=1)
        
    if "vww" in evals:
        _, _, vww_test = get_vww(model_cfg)
        vww_test_score = model.evaluate(vww_test, verbose=0)
        print(f"Visual Wake Words Test Score: {vww_test_score[1]}")
        result = pd.concat([result, pd.DataFrame({"vww_test_score": [vww_test_score[1]]})], axis=1)

    if "distance" in evals:
        dist_results = distance_eval(model, model_cfg)
        result = pd.concat([result, dist_results], axis=1)
    
    if "miap" in evals:
        miap_results = miap_eval(model, model_cfg)
        result = pd.concat([result, miap_results], axis=1)
        
    if "lighting" in evals:
        lighting_results = lighting_eval(model, model_cfg)
        result = pd.concat([result, lighting_results], axis=1)
        
    if "hands_feet" in evals:
        hands_feet_results = hands_feet_eval(model, model_cfg)
        result = pd.concat([result, hands_feet_results], axis=1)
        
    if "depiction" in evals:
        depiction_results = depiction_eval(model, model_cfg)
        result = pd.concat([result, depiction_results], axis=1)
        
    if "macs" in evals:
        macs_result = get_macs(model)
        result = pd.concat([result, macs_result], axis=1)

    print("Benchmark Complete")
    print(result)
    return result
    


if __name__ == "__main__":
    experiment_names = [
    # "image_1.5_long2024_03_04-03_54_13_PM",
    # "bbox_1.5_long_2024_03_04-03_53_52_PM",
    "wv_small_256x256_2024_03_04-10_58_34_AM",
    "wv_model_size_1.5_2024_03_03-11_41_34_PM",
    "wv_small_224x224_2024_03_03-09_53_43_PM",
    "wv_model_size_1.0_2024_03_03-10_10_51_AM",
    "wv_small_192x192_2024_03_03-07_15_34_AM",
    "wv_model_size_0.5_2024_03_02-10_31_25_PM",
    "wv_small_160x160_2024_03_02-06_32_44_PM",
    "wv_model_size_0.35_2024_03_02-10_26_32_AM",
    "wv_small_128x128_2024_03_02-01_56_12_AM",
    "wv_small_96x96_2024_03_01-01_27_44_PM",
    "wv_model_size_0.25_2024_03_01-01_20_49_PM",
    "wv_model_size_0.1_2024_02_29-09_45_09_PM",
    "wv_small_64x64_2024_02_29-09_40_58_PM",
    "grayscale_baseline_2024_02_28-08_12_29_PM",
    "wv_model_size_1.5_2024_02_25-03_18_41_PM",
    "wv_model_size_1.0_2024_02_25-03_13_21_AM",
    "wv_model_size_0.5_2024_02_24-03_08_14_PM",
    "wv_small_224x2242024_02_24-07_39_21_AM",
    "wv_model_size_0.35_2024_02_24-03_08_17_AM",
    "wv_small_192x1922024_02_23-06_06_32_PM",
    "vww_small_2024_02_23-04_08_49_PM",
    "wv_model_size_0.25_2024_02_23-02_22_02_PM",
    "wv_small_160x1602024_02_23-05_25_53_AM",
    "wv_model_size_0.1_2024_02_23-02_35_42_AM",
    "wv_small_2024_02_23-02_33_35_AM",
    "wv_small_2024_02_22-10_49_04_PM",
    "wv_small_128x1282024_02_22-10_14_29_AM",
    "wv_small_96x962024_02_21-06_19_59_PM",
    ]
    already_ran = pd.DataFrame()#pd.read_csv("full_benchmark_results.csv")
    
    
    results = pd.DataFrame()
    for model in experiment_names:
        # if model in already_ran["Experiment ID"].values:
        #     continue
        print(model)
        model_yaml = "gs://wake-vision-storage/saved_models/" + model + "/config.yaml"
        with tf.io.gfile.GFile(model_yaml, 'r') as fp:
            load_cfg = yaml.unsafe_load(fp)
        load_cfg = config_dict.ConfigDict(load_cfg)
        model_cfg = default_cfg.copy_and_resolve_references()
        model_cfg.update(load_cfg)
        
        benchmark_output = benchmark_suite(model_cfg)
        benchmark_output = pd.concat([pd.DataFrame({"Experiment ID": [model]}, benchmark_output)], axis=1)

        results = pd.concat([results, benchmark_output], ignore_index=True)
        results.to_csv("full_benchmark_results.csv")
        
    print("All Benchmarking Complete")
    print(results)
    
    results = pd.concat([already_ran, results], ignore_index=True)
    
    results.to_csv("full_benchmark_results.csv")
