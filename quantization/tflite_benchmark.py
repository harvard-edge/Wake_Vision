import numpy as np
import os
import pandas as pd
from experiment_config import default_cfg
import pandas as pd

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras

import tensorflow as tf
import tensorflow_datasets as tfds

from wake_vision_loader import get_distance_eval, get_wake_vision, get_lighting, get_miaps, get_depiction_eval
from vww_loader import get_vww

def evaluate(model, ds, batch_size=1, verbose=0):
    num_correct = 0
    num_batches = 0
    for image, label in ds:
        num_batches += 1
        probs = model(input_layer=image)["output_0"]
        num_correct += (tf.argmax(probs, axis=1).numpy() == label.numpy()).sum()
    if verbose:
        print(f"Accuracy: {num_correct / (num_batches * batch_size)}")
    return (0, num_correct / (num_batches * batch_size)) #tuple to match return pattern of keras evaluate
    

def f1(tp_rate, fp_rate, fn_rate):
    return 2 * tp_rate / (2 * tp_rate + fp_rate + fn_rate)

def lighting_eval(model, model_cfg=default_cfg):
    lighting_ds = get_lighting(model_cfg, batch_size=1)

    person_dim_score = evaluate(model, lighting_ds["person_dim"], verbose=0)
    person_normal_light_score = evaluate(model, lighting_ds["person_normal_light"], verbose=0)
    person_bright_score = evaluate(model, lighting_ds["person_bright"], verbose=0)
    
    non_person_dim_score = evaluate(model, lighting_ds["non_person_dim"], verbose=0)
    non_person_normal_light_score = evaluate(model, lighting_ds["non_person_normal_light"], verbose=0)
    non_person_bright_score = evaluate(model, lighting_ds["non_person_bright"], verbose=0)
    
    dim_f1 = f1(person_dim_score[1], 1-non_person_dim_score[1], 1-person_dim_score[1])
    normal_light_f1 = f1(person_normal_light_score[1], 1-non_person_normal_light_score[1], 1-person_normal_light_score[1])
    bright_f1 = f1(person_bright_score[1], 1-non_person_bright_score[1], 1-person_bright_score[1])
    
    result = pd.DataFrame({
        'lighting-dark': [dim_f1],
        'lighting-normal_light': [normal_light_f1],
        'lighting-bright': [bright_f1]})

    print(result)

    return result

def distance_eval(model, model_cfg=default_cfg):
    dist_cfg = model_cfg.copy_and_resolve_references()
    dist_cfg.MIN_BBOX_SIZE = 0.05 #ensure we always use the same min bbox size

    distance_ds = get_distance_eval(dist_cfg, batch_size=1)

    near_score = evaluate(model, distance_ds["near"], verbose=0)
    mid_score = evaluate(model, distance_ds["mid"], verbose=0)
    far_score = evaluate(model, distance_ds["far"], verbose=0)
    no_person_score = evaluate(model, distance_ds["no_person"], verbose=0)
    
    near_f1 = f1(near_score[1], 1-no_person_score[1], 1-far_score[1])
    mid_f1 = f1(mid_score[1], 1-no_person_score[1], 1-far_score[1])
    far_f1 = f1(far_score[1], 1-no_person_score[1], 1-far_score[1])

    result = pd.DataFrame({
        'distance-near': [near_f1],
        'distance-mid': [mid_f1],
        'distance-far': [far_f1]})
    print(result)

    return result

def miap_eval(model, model_cfg=default_cfg):
    miap_ds = get_miaps(model_cfg, batch_size=1)

    female_score = evaluate(model, miap_ds["female"], verbose=0)
    male_score = evaluate(model, miap_ds["male"], verbose=0)
    gender_unknown_score = evaluate(model, miap_ds["gender_unknown"], verbose=0)
    young_score = evaluate(model, miap_ds["young"], verbose=0)
    middle_score = evaluate(model, miap_ds["middle"], verbose=0)
    old_score = evaluate(model, miap_ds["older"], verbose=0)
    age_unknown_score = evaluate(model, miap_ds["age_unknown"], verbose=0)
    no_person_score = evaluate(model, miap_ds["no_person"], verbose=0)
    
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

def depiction_eval(model, model_cfg=default_cfg):
    depiction_ds = get_depiction_eval(model_cfg, batch_size=1)

    person_score = evaluate(model, depiction_ds["person"], verbose=0)
    depictions_persons_score = evaluate(model, depiction_ds["depictions_persons"], verbose=0)
    depictions_non_persons_score = evaluate(model, depiction_ds["depictions_non_persons"], verbose=0)
    non_person_no_depictions_score = evaluate(model, depiction_ds["non_person_no_depictions"], verbose=0)
    
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


def tflite_benchmark_suite(model, evals=["vww", "distance", "miap", "lighting", "depiction"]):

    
    _, _, wv_test = get_wake_vision(batch_size=1)
    wv_test_score = evaluate(model, wv_test, verbose=0)
    print(f"Wake Vision Test Score: {wv_test_score[1]}")
    result = pd.DataFrame({"wv_test_score": [wv_test_score[1]]})
        
    if "vww" in evals:
        _, _, vww_test = get_vww(batch_size=1)
        vww_test_score = evaluate(model, vww_test, verbose=0)
        print(f"Visual Wake Words Test Score: {vww_test_score[1]}")
        result = pd.concat([result, pd.DataFrame({"vww_test_score": [vww_test_score[1]]})], axis=1)

    if "distance" in evals:
        dist_results = distance_eval(model)
        result = pd.concat([result, dist_results], axis=1)
    
    if "miap" in evals:
        miap_results = miap_eval(model)
        result = pd.concat([result, miap_results], axis=1)
        
    if "lighting" in evals:
        lighting_results = lighting_eval(model)
        result = pd.concat([result, lighting_results], axis=1)
        
    if "depiction" in evals:
        depiction_results = depiction_eval(model)
        result = pd.concat([result, depiction_results], axis=1)

    print("Benchmark Complete")
    print(result)
    return result
    


if __name__ == "__main__":
    tflite_model_path = "base_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    model = interpreter.get_signature_runner()
    

    results = tflite_benchmark_suite(model)
        
    print("All Benchmarking Complete")
    print(results)
    
    results.to_csv("benchmark_results_tflite.csv")
