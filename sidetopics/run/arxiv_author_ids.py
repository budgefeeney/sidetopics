#!/opt/local/bin/python3.4

import sys
import os
import re


def run(infile, outfile):
    authors = set()
    with open(infile, "r") as f:
        for line in f:
            if line[:6] == "author":
                core = line.split("{")[1]
                core = core.split("}")[0].strip()
                names = core.split(";")
                for name in names:
                    authors.add(name.strip())

    author_list = [a for a in authors]
    author_list.sort()
    i = 1
    with open(outfile, "w") as f:
        for author in author_list:
            f.write("%d\t%s\n" % (i, author))
            i += 1



if __name__ == '__main__':
    prefix="/Users/bryanfeeney/Downloads/cit-hepth/release/2013/"
    run(prefix + "acl-metadata.txt", prefix + "author_ids.txt")
