import tensorflow_datasets as tfds


builder = tfds.list_builders()
print(builder)
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']
print(test_data)

