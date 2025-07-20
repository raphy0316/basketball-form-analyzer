import json
import os

END_FRAME = 232350

bench_mark_phases = []
correct_count = 0
total_count = 0
correct_seq_count = 0
total_seq_count = 0

def load_phase_ranges(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if(line == "\n"):
                continue
            seg, label = line.strip().split(' ')
            start, end = map(int, seg.split("~"))
            if end > END_FRAME:
                continue

            global total_seq_count
            if label == "Release":
                total_seq_count += 1

            bench_mark_phases.append({
                "start": start,
                "end": end,
                "label": label
            })

def load_json_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def compare_phases(data):

    found_idx = -2 # found index of the last release phase

    for frame in data["frames"]:
        idx = frame.get("frame_index", -1)
        if idx < 0 or idx > END_FRAME:
            continue
        
        pose = frame.get("normalized_pose", {})
        if not pose:
            continue
        frame_phase = frame.get("phase", "General")
        if frame_phase not in {"Rising", "General", "Follow-through", "Release"}:
            continue
        
        phase_label = "General"
        phase_start, phase_end = -1, -1
        # Can be optmized with two pointers

        release_idx = -1 # index of the last release phase
        for i, phase in enumerate(bench_mark_phases):
            if phase["start"] <= idx <= phase["end"]:
                phase_label = phase["label"]
                phase_start = phase["start"]
                phase_end = phase["end"]
                if phase_label == "Release":
                    release_idx = i
                break
        
        # Check if the current frame is within the release phase
        global correct_seq_count
        if phase_label == "Release" and (idx <= phase_end + 5 and idx >= phase_start - 5) and frame_phase == "Release" and found_idx != release_idx:
            correct_seq_count += 1
            found_idx = release_idx
        
        # Check if the phase matches the frame's phase
        global correct_count, total_count
        total_count += 1
        if phase_label == frame_phase:
            correct_count += 1

def main():
    # Define paths to relevant directories
    labels_dir = "../labels"
    results_dir = "../../data/results"
    
    # List available files in each directory
    print("Available label files in ../labels:")
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    for i, file in enumerate(labels_files):
        print(f"  [{i}] {file}")
    
    # Get user selection
    label_idx = int(input("\nSelect label file (number): "))

    print("\nAvailable result files in ../data/results:")
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    for i, file in enumerate(result_files):
        print(f"  [{i}] {file}")
    

    result_idx = int(input("Select result file (number): "))
    
    # Construct full paths
    txt_file = os.path.join(labels_dir, labels_files[label_idx])
    json_file = os.path.join(results_dir, result_files[result_idx])
    
    print(f"\nUsing:\n- Label file: {txt_file}\n- Result file: {json_file}")
    
    if not os.path.exists(txt_file) or not os.path.exists(json_file):
        print("Required files not found.")
        return
    
    load_phase_ranges(txt_file)
    data = load_json_data(json_file)
    compare_phases(data)

    print(f"Correct: {correct_count}, Total: {total_count}, Accuracy: {correct_count / total_count if total_count > 0 else 0:.2f}")
    print(f"Correct Release Sequences: {correct_seq_count}, Total Release Sequences: {total_seq_count}, Release Sequence Accuracy: {correct_seq_count / total_seq_count if total_seq_count > 0 else 0:.2f}")

if __name__ == "__main__":
    main()