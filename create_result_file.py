import os
import argparse

if __name__ == '__main__':

    # Parent Directory path
    parent_dir = "./RESULTS"

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)   
        print('RESULTS file created.') 

    parser = argparse.ArgumentParser(description='Create File')
    parser.add_argument('--file_name', type=str, required=True, default='new_file', help='file name')
    args = parser.parse_args()

    # Directory
    directory = args.file_name
    
    # Parent Directory path
    parent_dir = "./RESULTS"
    
    # Path
    path = os.path.join(parent_dir, directory)

    if os.path.exists(path):
        print("Directory '% s' exists" % directory)
    else:
        os.mkdir(path)
        print("Directory '% s' created" % directory)