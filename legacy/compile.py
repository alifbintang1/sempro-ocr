import os

output_file = "all_python_files.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        if "venv" in root:
            continue  # Lewati direktori virtual environment
        print(f"Memproses direktori: {root}")
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                print(f"Memproses file: {filepath}")
                outfile.write(f"\n--- FILE: {filepath} ---\n\n")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"[ERROR reading file: {e}]\n\n")

print(f"Selesai. Semua file .py digabung ke {output_file}")