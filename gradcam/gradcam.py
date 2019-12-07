from skimage import io
import cv2
from keras.preprocessing import image
from keras import backend as K
from keras.applications.vgg16 import preprocess_input

def gradcam_cal(inputImage,layer,inputmodel):
  inputImage = io.imread(inputImage)
  converted = cv2.resize(inputImage, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
  x = image.img_to_array(converted)
  x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  model = inputmodel
  preds = model.predict(x)
  class_idx = np.argmax(preds[0])
  print(class_idx)
  class_output = model.output[:, class_idx]
  last_conv_layer = model.get_layer(layer)
  grads = K.gradients(class_output, last_conv_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
  pooled_grads_value, conv_layer_output_value = iterate([x])
  for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  heatmap = np.mean(conv_layer_output_value, axis = -1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap = cv2.resize(heatmap, (converted.shape[1], converted.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = cv2.addWeighted(converted, 0.5, heatmap, 0.5, 0)
  return converted, superimposed_img
