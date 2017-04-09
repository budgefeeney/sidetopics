#!/opt/local/bin/python3.4

import sys
import os
import re

LatexEscape = re.compile(r"\\['`^" + '"' + r"H~ckl=b.druvtoi]")
Dashes = re.compile(r"^-{15,}$")
StartsWithWhiteSpace = re.compile(r"^\s{2,}\S.*$")
BibTextNameDelimiter = re.compile(r"(?:\s*,\s*|\s+and\s+)")

class PaperMetadata():

    def __init__(self, arxiv_id=-1, title=None, authors=None, venue=None, year=-1):
        self.arxiv_id = arxiv_id
        self.title    = title
        self.authors  = [] if authors is None else authors
        self.venue    = venue
        self.year     = year

    def complete(self):
        '''
        Returns true if all necessary fields have been read in,
        false otherwise
        '''
        return self.arxiv_id > 0 \
            and self.title is not None \
            and len(self.authors) > 0 \
            and self.venue is not None \
            and self.year > 0

    def complete_except_for_venue(self):
        '''
        Returns true if all necessary fields have been read in,
        with the exception of venue, false if not.
        '''
        return self.arxiv_id > 0 \
            and self.title is not None \
            and len(self.authors) > 0 \
            and self.year > 0


def unlatex(latexStr):
    '''
    Remove Latex accent commands
    '''
    return LatexEscape.sub("", latexStr)


def bibtex_to_acl_author (bib_author):
    '''
    Takes an author name formated in the BibTex style, and re-write
    it in the ACL style
    '''
    parts = bib_author.strip().split(" ")

    if len(parts) == 1:
        parts = parts[0].split(".")
        if len(parts) == 1:
            return parts[0]
        else:
            return parts[-1] + ", " + ". ".join(parts[0:-1]) + "."
    elif len(parts) == 2:
        return parts[-1] + ", " + parts[0]
    elif len(parts) >= 3:
        if parts[-1].lower() in ["i", "ii", "iii", "jr", "jr.", "sr", "sr.", "phd", "phd."]:
            parts[-2] = parts[-2] + " " + parts[-1]
            parts.pop()
        if len(parts) >= 3 and parts[-2].lower() in ["mc", "mac", "o", "d'", "de", "del", "della", "di", "van", "von"]:
            return parts[-2] + " " + parts[-1] + ", " + " ".join(parts[0:-2])
        else:
            return parts[-1] + ", " + " ".join(parts[0:-1])


def write_paper_metadata(f, paper):
    '''
    Writes an entry to the given file in the same format as
    the ACL metadata file

    :param f: the open writeable file object to write to
    :param paper: the PaperMetadata object with the paper's metadata
    '''
    def stripBraces(val):
        return val.translate(str.maketrans('{}','()'))

    f.write("id={%d}\n"     % paper.arxiv_id)
    f.write("title={%s}\n"  % stripBraces(paper.title))
    f.write("author={%s}\n" % stripBraces("; ".join(bibtex_to_acl_author(a) for a in paper.authors)))
    if paper.venue is not None:
        f.write("venue={%s}\n"  % stripBraces(paper.venue))
    f.write("year={%d}\n"   % paper.year)
    f.write("\n")


def read_next_key_value(lines):
    '''
    Reads the next key-value pair form the given file contents. Keys and
    values are delimited by colons. Values may extend to the next line
    if the next line begins with two or more white-space characters,
    but contains at least one non-whitespace character.

    This automatically skips over comments, blank lines, etc, till the next
    key-value pair is found.

    The key and value are trimmed. The key is lower-cased.

    If we reach the end of the file, None, None is returned.

    :param lines: the file contents as a list of strings, each corresponding
    to an untrimmed line inthe file (i.e. file.readlines())
    :return: a key-value pair, both trimmed, and the key-lowercased. Returns
    None,None if the end of hte file has been reached.
    '''
    AbstractMarker = r"\\"
    def is_skippable(line):
        return line == "" or Dashes.match(line)

    # Skip empty lines
    line = ""
    while is_skippable(line) and len(lines) > 0:
        line = lines.pop(0).strip()
    if is_skippable(line): # end of file
        return None, None
    if line == AbstractMarker: # section marker, abstract follows
        return None, None

    parts = line.split(":", maxsplit=1)
    key, value = parts[0].strip().lower(), parts[1].strip()

    # See if the value continues onto other lines
    while len(lines) > 0 \
            and lines[0] != AbstractMarker\
            and StartsWithWhiteSpace.match(lines[0]):
        value = value + " " + lines.pop(0).strip()

    return key, value


def eliminateParentheses(value):
    '''
    Eliminate anything in parentheses, include the parentheses themselves
    :param value:
    :return:
    '''
    out = ""
    inside = 0 # how far nested into parentheses are we
    for i in range(len(value)):
        if inside > 0:
            if value[i] == ")":
                inside -= 1
            elif value[i] == "(":
                inside += 1
        else: # outside
            if value[i] == "(":
                inside = 1
            else:
                out += value[i]

    return out


def read_paper_metadata(path):
    '''
    Reads the metadata from the file at the given path and returns a
    PaperMetadata object with the metadata of the paper stored in that
    file.

    It's assumed the filename is the paper's arxiv ID plus ".abs", and that
    the first two characters of that name provide the (two-digit) year of
    submission.

    '''
    fname    = os.path.split(path)[1]
    arxiv_id = int(fname[0:-4])
    year     = int(fname[:2])
    year     = year + 1900 if year >= 90 else year + 2000

    print ("Translating paper with ID " + fname[0:-4])

    data = PaperMetadata(arxiv_id=arxiv_id, year=year)

    with open(path, "r") as f:
        lines = f.readlines()
    lines = lines[2:] # skip dashed line and first section marker

    while not data.complete():
        key, value = read_next_key_value(lines)
        if key is None:
            break

        if key == "author" or key == "authors":
            value = eliminateParentheses(value)
            data.authors = [a.strip() for a in BibTextNameDelimiter.split(value) if len(a.strip()) > 0]
        elif key == "title":
            data.title = value
        elif key == "journal-ref":
            data.venue = value.strip().split(" ")[0]

    if not data.complete_except_for_venue():
        raise ValueError("Incomplete record for file ID %d" % data.arxiv_id)

    return data


def run(args):
    assert len(args) == 2, "Need to specify input file and output file"

    paper = read_paper_metadata(args[0])

    if args[1] == "-":
        write_paper_metadata (sys.stdout, paper)
    else:
        with open(args[1], "w") as f_out:
            write_paper_metadata (f_out, paper)


def run_on_dirs():
    parent_dir = "/Users/bryanfeeney/Downloads/cit-HepTh-abstracts"
    with open("/Users/bryanfeeney/Desktop/arxiv-metadata.txt", "w") as f:
        for year in [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003]:
            files_dir = parent_dir + "/" + str(year)
            infiles = os.listdir(files_dir)
            for infile in infiles:
                if not infile[-4:] == ".abs":
                    continue
                try:
                    paper = read_paper_metadata(files_dir + "/" + infile)
                    write_paper_metadata(f, paper)
                except ValueError as e:
                    print ("Skipping entry for file " + infile)
                    print (str(e))


if __name__ == '__main__':
    # run(args=sys.argv[1:])
    # run(["/Users/bryanfeeney/Downloads/cit-HepTh-abstracts/1996/9601091.abs", "-"])
    run_on_dirs()
