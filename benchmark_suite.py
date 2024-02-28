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


def benchmark_suite(model_cfg, evals=["test", "distance", "miap", "lighting", "hands_feet", "depiction"]):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)
    
    result = pd.DataFrame({'model': [model_cfg.MODEL_NAME]})

    if "test" in evals:
        _, _, wv_test = get_wake_vision(model_cfg)
        wv_test_score = model.evaluate(wv_test, verbose=0)
        print(f"Wake Vision Test Score: {wv_test_score[1]}")
        result = pd.concat([result, pd.DataFrame(wv_test_score, index=["wv_test_score"])], axis=1)

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

    print("Benchmark Complete")
    print(result)
    return result
    


if __name__ == "__main__":
    experiment_names = [
    "wv_model_size_0.1_2024_02_23-02_35_42_AM",
    "wv_model_size_0.25_2024_02_23-02_22_02_PM",
    "wv_model_size_0.35_2024_02_24-03_08_17_AM",
    "wv_model_size_0.5_2024_02_24-03_08_14_PM",
    "wv_model_size_1.0_2024_02_25-03_13_21_AM",
    "wv_model_size_1.5_2024_02_25-03_18_41_PM",
    ]
    results = pd.DataFrame()
    for model in experiment_names:
        print(model)
        model_yaml = "gs://wake-vision-storage/saved_models/" + model + "/config.yaml"
        with tf.io.gfile.GFile(model_yaml, 'r') as fp:
            load_cfg = yaml.unsafe_load(fp)
        load_cfg = config_dict.ConfigDict(load_cfg)
        model_cfg = default_cfg.copy_and_resolve_references()
        model_cfg.update(load_cfg)
            

        results = pd.concat([results, benchmark_suite(model_cfg)], ignore_index=True)
        
    print("All Benchmarking Complete")
    print(results)
    
    results.to_csv("benchmark_results.csv")