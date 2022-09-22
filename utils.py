import numpy as np

def cutArea(img_file, result_predict):
  result_predict[result_predict > 0 ] = 255
  result_predict[result_predict < 255 ] = 0
  result_predict = result_predict.astype(np.uint8)

  return np.multiply(img_file, np.where(result_predict == 0, result_predict, 1))

def cutAreaInverted(img_file, result_predict):
  result_predict[result_predict == 0 ] = 255
  result_predict[result_predict != 255 ] = 0
  result_predict = result_predict.astype(np.uint8)

  return np.multiply(img_file, np.where(result_predict == 0, result_predict, 1))

def makeImageMask(result_predict):
    result_predict[result_predict > 0 ] = 255
    result_predict[result_predict < 255 ] = 0
    result_predict = result_predict.astype(np.uint8)
    return result_predict

def makeImageMaskInverted(result_predict):
  result_predict[result_predict == 0 ] = 255
  result_predict[result_predict != 255 ] = 0
  result_predict = result_predict.astype(np.uint8)
  return result_predict