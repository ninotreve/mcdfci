import sys
import json

system = sys.argv[1]
N = int(sys.argv[2])
path = sys.argv[3]

with open("{}/{}.json".format(path, system)) as f:
    d = json.load(f)
dsub = d["solver"]["cdfci"]
dsub.update({"num_of_coordinate": N, "estimate": N})
dsub["num_iterations"] = int(dsub["num_iterations"] / N)
dsub["report_interval"] = int(dsub["report_interval"] / N)
with open("{}/{}_{}.json".format(path, system, N), 'w') as fin:
    json.dump(d, fin, indent = 4)
