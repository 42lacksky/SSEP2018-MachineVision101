import pickle


def load_model():
    with open("knn.pkl", "rb") as f:
        model = pickle.load(f)

    return model
