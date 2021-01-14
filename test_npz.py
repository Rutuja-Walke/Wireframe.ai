import numpy as np
file_path="./resources/data/0A9E9E2A-E7FB-4462-890A-CEB77C99D148.npz"
retrieve = np.load(file_path)["features"]
print(retrieve.shape)