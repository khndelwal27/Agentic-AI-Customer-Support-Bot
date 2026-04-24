import pickle

model = pickle.load(open("model.pkl", "rb"))

print("\n✅ Model Intent Classes:\n")
print(model.classes_)
