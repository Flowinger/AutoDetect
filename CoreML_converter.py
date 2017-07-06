import coremltools
from keras.models import load_model

print('Model filepath:')
model_input = input('>')
model = load_model(model_input)

scale = 1./255
coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names='image',
                                                    #output_names=['probs'],
                                                    image_input_names='image',
                                                    class_labels='classes.txt',
                                                    predicted_feature_name='class',
                                                    image_scale=scale)

# Save Core ML Model
coreml_model.author = 'FP'
coreml_model.save("autodetect_v3.mlmodel")