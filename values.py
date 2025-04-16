import numpy as np
import json

# Parameters
low = 1.0e-5
high = 0.1
size = 25  # Number of values to generate

# Generate log-uniform values using logspace
values = np.logspace(np.log10(low), np.log10(high), size)

# Convert to list (as numpy arrays aren't JSON serializable) and print as JSON
print(json.dumps(values.tolist()))
