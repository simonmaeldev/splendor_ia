import os

def count_files_and_lines(directories):
    for dir in directories:
        total_files = 0
        total_lines = 0
        for root, _, files in os.walk(dir):
            for file in files:
                total_files += 1
                try:
                    with open(os.path.join(root,file), 'r', errors='ignore') as f:
                        total_lines += sum(1 for _ in f) - 1
                except Exception as e:
                    print(f"file {root}/{file} failed with {e}")
                    continue
        print(f"dir: {dir}")
        print(f"total files: {total_files}")
        print(f"total lines: {total_lines}")
    return total_files, total_lines

dirs = ["data/games/2_games", "data/games/3_games", "data/games/4_games"]
count_files_and_lines(dirs)

