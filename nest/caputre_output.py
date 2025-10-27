import subprocess
import re
import matplotlib.pyplot as plt

# 1️⃣ Run the main simulation script
result = subprocess.run(
    ["python3", "test.py"],
    capture_output=True,
    text=True,
    bufsize=1,  # line-buffered
)

# Combine stdout + stderr
output = result.stdout + result.stderr

# 2️⃣ Regex to capture debug info
debug_pattern = re.compile(
    r"\[DEBUG trigger_update_weight\]\s*\|\s*post_node_id=(\d+)\s*\|\s*t_trig=(\S+)\s*\|.*?c_delayed=(\S+)"
)

debug_data = [
    (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    for m in debug_pattern.finditer(output)
]
# print(debug_data)

# 3️⃣ Filter for post_neuron_id = 66
debug_data_66 = [ (t, c) for post_id, t, c in debug_data if post_id == 19 ]

print(f"Collected {len(debug_data_66)} debug lines for neuron 66")

# 4️⃣ Plot
if debug_data_66:
    time_points, c_delayed_values = zip(*debug_data_66)
    # print(time_points, c_delayed_values)

    plt.figure(figsize=(10, 4))
    plt.plot(time_points, c_delayed_values, linestyle='-', color='purple')
    plt.xlabel("Time (ms)")
    plt.ylabel("c_delayed")
    plt.title("Eligibility trace c_delayed for post neuron 66")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No c_delayed entries found for post neuron 66")
