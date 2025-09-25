# import os

# def count_images(folder_path):
#     # common image extensions
#     image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    
#     count = 0
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if os.path.splitext(file)[1].lower() in image_extensions:
#                 count += 1
#     return count

# if __name__ == "__main__":
#     folder = "/data/data/shibu/PiT/inference/vehicles_3"
#     if os.path.isdir(folder):
#         total = count_images(folder)
#         print(f"Total images in '{folder}': {total}")
#     else:
#         print("Invalid folder path!")

import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip broken symlinks
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(bytes_size):
    # convert bytes to human-readable units
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024

if __name__ == "__main__":
    folder =  "/data/data/shibu/PiT/inference/animals_10"
    if os.path.isdir(folder):
        size_bytes = get_folder_size(folder)
        print(f"Total size of '{folder}': {format_size(size_bytes)}")
    else:
        print("Invalid folder path!")
