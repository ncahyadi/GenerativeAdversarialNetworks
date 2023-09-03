import os

def info_dir():
    current_dir = os.getcwd()
    print(f'Current working directory is: {current_dir}')
    return current_dir

def checking_dir_existance(path):
    existance = os.path.isdir(path)
    print(f'The directory is existed' if os.path.isdir(path) else f'The directory is not existed')
    return existance

def create_dir(parent_dir, new_childir):
    new_path = os.path.join(parent_dir, new_childir)
    existance = checking_dir_existance(new_path)
    print(f'Creating no directories') if existance else (print(f'Creating new directories {new_path}'), os.mkdir(new_path))[1]
    return new_path

