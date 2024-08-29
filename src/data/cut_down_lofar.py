import os
import pickle


def main(limit: float):
    filepath = os.path.join("./data", "LOFAR_Full_RFI_dataset.pkl")
    with open(filepath, "rb") as f:
        train_x, train_y, test_x, test_y = pickle.load(f)
    ilimit = int(limit * len(train_x))
    train_x = train_x[:ilimit]
    train_y = train_y[:ilimit]
    with open(f"./data/LOFAR_{limit}_RFI_dataset.pkl", "wb") as f:
        pickle.dump((train_x, train_y, test_x, test_y), f)


if __name__ == "__main__":
    limit = 0.1
    main(limit)
