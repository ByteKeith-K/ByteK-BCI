import scipy.io
import os
import glob

MAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stroke_mat")


def inspect_mat_file():
    # Search for the first .mat file
    mat_files = glob.glob(os.path.join(MAT_DIR, "**", "*.mat"), recursive=True)

    if not mat_files:
        print("No .mat files found in data/stroke_mat/. Did you extract them there?")
        return

    first_file = mat_files[0]
    print(f"Loading {first_file}...\n")

    # Load the MATLAB file
    mat_data = scipy.io.loadmat(first_file)

    print("--- VARIABLES INSIDE THE .MAT FILE ---")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"Variable Name: '{key}'")
            print(f"Data Type: {type(mat_data[key])}")

            # If it's a numpy array, show its shape
            if hasattr(mat_data[key], 'shape'):
                print(f"Shape: {mat_data[key].shape}")

                # If it's small enough, print the first few values
                if mat_data[key].size < 20:
                    print(f"Values: {mat_data[key]}")

            print("-" * 40)


if __name__ == "__main__":
    inspect_mat_file()