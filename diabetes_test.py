import pickle
import numpy as np

# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\HP\Documents\diabetes_model.sav", 'rb'))


input_data = (6,148	,72	,35	,0,	33.6,0.627,50)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a diabetes')
else:
  print('The Person has diabetes')