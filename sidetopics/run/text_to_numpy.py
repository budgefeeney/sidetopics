import os
import sys
import numpy as np
import pickle as pkl
import scipy.sparse as sp

# InputDir="/Users/bryanfeeney/Downloads/NUS_WID_Low_Level_Features/Low_Level_Features"
InputDir="/Users/bryanfeeney/Downloads/NUS_WID_Low_Level_Features/tags"

InputFileExt=".dat"
InputFileLineCountExt=".cnt"
OutFileExt=".npy"

def die(errmsg):
    print (errmsg)
    exit(-1)

def save_as_numpy_sparse (infile, row_count):
    with open (infile, "r") as f:
        first_line = f.readline()
        col_count = len(first_line.split())
        dtype = np.float32 if "." in first_line else np.int32

    matrix = sp.lil_matrix((row_count, col_count), dtype=dtype)
    row = 0
    with open(infile, "rb") as f:
        for line in f:
            try:
                vals = [float(fstr) for fstr in line.split()]
                for (col, val) in enumerate(vals):
                    if val > 0:
                        matrix[row, col] = val
            except ValueError as e:
                print ("Can't read line " + str(row) + " from file " + str(infile) + " " + str(line))
                die(str(e))

            row += 1
    if row != row_count:
        die ("Rows read is %d but expected row-count is %d" % (row, row_count))

    outfile = infile[:-4] + OutFileExt
    with open(outfile, "wb") as f:
        pkl.dump(matrix.tocsr(), f)

def save_as_numpy (infile, row_count):
    with open (infile, "r") as f:
        first_line = f.readline()
        col_count = len(first_line.split())
        dtype = np.float32 if "." in first_line else np.int32

    matrix = np.ndarray(shape=(row_count, col_count), dtype=dtype)
    row = 0
    with open(infile, "rb") as f:
        for line in f:
            try:
                vals = [float(fstr) for fstr in line.split()]
            except ValueError as e:
                print ("Can't read line " + str(row) + " from file " + str(infile) + " " + str(line))
                die(str(e))

            matrix[row,:] = vals
            row += 1
    if row != row_count:
        die ("Rows read is %d but expected row-count is %d" % (row, row_count))

    outfile = infile[:-4] + OutFileExt
    with open(outfile, "wb") as f:
        pkl.dump(matrix, f)


def read_wc_l_output(infile):
    with open(infile, "r") as f:
        line = f.readline()
        parts = line.split()
        return int(parts[0])

def run(args):
    # for file in os.listdir(InputDir):
    #     if file.endswith(InputFileExt):
    #         line_count = read_wc_l_output(InputDir + "/" + file + InputFileLineCountExt)
    #         save_as_numpy(InputDir + "/" + file, line_count)
    for file in ["AllTags1k.txt", "AllTags81.txt", "Test_Tags1k.dat", "Test_Tags81.txt", "Train_Tags1k.dat", "Train_Tags81.txt"]:
        line_count = read_wc_l_output(InputDir + "/" + file + InputFileLineCountExt)
        save_as_numpy(InputDir + "/" + file, line_count)

if __name__ == '__main__':
    run(args=sys.argv[1:])

