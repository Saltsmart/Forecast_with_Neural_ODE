import os
import shutil
import argparse
parser = argparse.ArgumentParser(description="Delect results and checkpoint files of these IDs")
parser.add_argument("--ID", nargs="+", default=None)
parser.add_argument("--data", nargs="+", default=None)

def find_and_del_path(ID_list, dir):
    path = os.path.join(os.getcwd(), dir)
    files = os.listdir(path)
    for file_name in files:
        check_child = True
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            for ID_item in ID_list:
                if ID_item in file_name:
                    check_child = False
                    shutil.rmtree(file_path)
                    break
            if check_child:
                find_and_del_path(ID_list, file_path)


def find_and_del_file(ext, dir_list):
    path = os.path.join(os.getcwd(), "data")
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        if not os.path.isdir(dir_path):
            continue
        files = os.listdir(dir_path)
        for file_name in files:
            if os.path.splitext(file_name)[-1] == ext:
                os.remove(os.path.join(dir_path, file_name))


if __name__ == "__main__":
    args = parser.parse_args()
    args.data = ["GoogleStock"]
    if args.ID is not None:
        print("Delect results and checkpoint files of these IDs:")
        print(args.ID)
        find_and_del_path(args.ID, "fig")
        find_and_del_path(args.ID, "log")
        find_and_del_path(args.ID, "ckpt")
    elif args.data is not None:
        print("Delect preprocessing files of these datasets:")
        print(args.data)
        find_and_del_file(".pkl", args.data)
    else:
        print("Can't figure out what to do. exit.")
