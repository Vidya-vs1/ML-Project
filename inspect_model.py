import pickle

with open("models/slang_classifier.pkl", "rb") as f:
    model = pickle.load(f)

print(f"🔍 Top-level type: {type(model)}")
print("📦 Top-level keys:", model.keys())

print("\n--- slang_to_standard ---")
print(f"Type: {type(model['slang_to_standard'])}")
print(model['slang_to_standard'])

print("\n--- standard_to_slang ---")
print(f"Type: {type(model['standard_to_slang'])}")
print(model['standard_to_slang'])
