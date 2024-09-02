import os

folders_to_check = ["Data", "Results", "Events"]

for folder in folders_to_check:
    folder_path = os.path.join(os.getcwd(), folder)
    
    if not os.path.exists(folder_path):
        print(f"'{folder}' folder not found. Creating the folder.")
        os.makedirs(folder_path)
    else:
        print(f"'{folder}' folder already exists.")

events_dir = os.path.join(os.getcwd(), "Events")
os.chdir(events_dir)

subfolders = ["Uniform", "Flagship"]

for folder in subfolders:
    if not os.path.exists(folder):
        print(f"'{folder}' folder not found. Creating the folder.")
        os.makedirs(folder)
    else:
        print(f"'{folder}' folder already exists.")
