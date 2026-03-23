import pickle
import numpy as np


p = "tables/q_table.pkl"
with open(p, "rb") as f:
    q_dict = pickle.load(f)

if not isinstance(q_dict, dict):
    print("Loaded object is not a dict:", type(q_dict))
else:
    print(f"Loaded dict with {len(q_dict)} entries\n")
    # How many entries to print (set to None to print all)
    N = 20
    for i, (k, v) in enumerate(q_dict.items()):
        if N is not None and i >= N:
            break
        print(f"Entry {i+1} — key: {k}")
        if isinstance(v, (list, tuple)):
            print(" value (sequence):", v[:10])
        elif isinstance(v, (np.ndarray,)) or hasattr(v, "shape"):
            arr = np.asarray(v)
            print(" value (numpy) shape:", arr.shape)
            print("  values:", arr.tolist())
        else:
            print(" value type:", type(v))
    if N is not None and len(q_dict) > N:
        print(f"\n... {len(q_dict)-N} more entries not shown")
