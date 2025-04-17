import tensorflow as tf

#path_to_test_set = '../../datasets/visual_wake_words/mini_val'
path_to_test_set = '../../datasets/wake_vision/wake_vision/test'

#image_size = (224,224)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_mobilenetv2.tflite'

image_size = (50,50)
color_mode = 'grayscale'
path_to_tflite_model = '0_wv_quality_micronets_vww2_50_50_INT8.tflite'

#image_size = (128, 128)
#color_mode = 'grayscale'
#path_to_tflite_model = 'wv_quality_micronets_vww3_128_128_INT8.tflite'
#path_to_tflite_model = 'wv_quality_micronets_vww4_128_128_INT8.tflite'

#image_size = (64, 64)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_mcunet-10fps_vww.tflite'

#image_size = (80, 80)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_mcunet-5fps_vww.tflite'

#image_size = (144, 144)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_mcunet-320kb-1mb_vww.tflite'

#image_size = (50,50)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_k_2_c_3.tflite'
#path_to_tflite_model = 'wv_quality_k_4_c_5.tflite'
#path_to_tflite_model = 'wv_quality_k_8_c_5.tflite'

#image_size = (50,50)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_ymac.tflite'

#image_size = (80,80)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_samy.tflite'

#image_size = (50,50)
#color_mode = 'rgb'
#path_to_tflite_model = '0_wv_quality_anas_benalla.tflite'

#image_size = (80,80)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_mohammad_hallaq.tflite'

#image_size = (50,50)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_apighetti.tflite'

#image_size = (48,48)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_benx13.tflite'

#image_size = (50,50)
#color_mode = 'rgb'
#path_to_tflite_model = 'wv_quality_cezar.tflite'

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_test_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=1,
    image_size=image_size,
    shuffle=True,
    seed=11,
    )
   
interpreter = tf.lite.Interpreter(path_to_tflite_model)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       pred = interpreter.get_tensor(output['index'])
       if 1 < pred.shape[1] :
           if label.numpy().argmax() == pred.argmax() :
               correct = correct + 1
           else :
               wrong = wrong + 1
       elif 1 == pred.shape[1] :
           pred = 1 if (pred / 255. >= .5) else 0
           if label.numpy().argmax() == pred:
               correct = correct + 1
           else:
               wrong = wrong + 1

print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
