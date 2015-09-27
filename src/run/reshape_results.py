# -*- coding: utf-8 -*-
'''
Reshapes the results from a precision & recall at K run
'''
__author__ = 'bryanfeeney'

import sys

All, Range0_3, Range3_up, Range3_5, Range5_up, Range5_10, Range10_up = \
    "0->", "0->3", "3->", "3->5", "5->", "5->10", "10->"
FullGroups=[All, Range0_3, Range3_up, Range3_5, Range5_up, Range5_10, Range10_up]
FileGroups=[All, Range0_3, Range3_5, Range5_10, Range10_up]
Ms = ["10", "20", "30", "40", "50", "75", "100", "150", "250", "500"]
DocCount = "DocCount"
TableTitles=[ DocCount ] + Ms
PerplexityPrefix="Training perplexity is "
PrecisionTitle="Precision"
RecallTitle="Recall"
MrrPrefix="Mean reciprocal-rank : "
MapPrefix="Mean Average Precision : "

DefaultInPath="/Users/bryanfeeney/iCloud/Results/ACL/Results Compendium.txt"
DefaultOutPath="/Users/bryanfeeney/iCloud/Results/ACL/ResultsCompendium.csv"




class Results:
    def __init__ (self, precs, recs, map, mrr, perp):
        self.precs = precs
        self.recs  = recs
        self.map   = map
        self.mrr   = mrr
        self.perp  = perp

class Algorithm:
    def __init__ (self, algorithm, topicCount, trainingDescription):
        self.algorithm = algorithm
        self.topicCount = topicCount
        self.trainingDescription = trainingDescription

    @classmethod
    def from_file_entry(cls, line):
        parts = line.split(" ")

        return Algorithm(
            parts[0].strip(" \t\r\n,"),
            parts[1].strip(" \t\r\n,K="),
            "" if len(parts) == 2 else " ".join(parts[2:]).strip()
        )

    def __str__(self):
        return self.algorithm + ", K=" + self.topicCount + " " + self.trainingDescription

    def to_delim_str(self, delim="\t", quoteFunc=id):
        return quoteFunc(self.algorithm) + delim + self.topicCount + delim + quoteFunc(self.trainingDescription)


def new_group_k_dict():
    return { g:{ k:0.0 for k in TableTitles } for g in FullGroups }


def read_next_result_batch(line, file):
    '''
    Reads results from a file, into a <code>Results</code> object
    and returns it. We presume the algorithm title has already
    been read.
    :param file: the file object to read from
    :return: a Results object
    '''

    # Perplexity (not all readings have this)
    perp,  line = read_perplexity(line, file)
    precs, line = read_precs(line, file)
    recs,  line = read_recs(line, file)
    mrr,   line = read_mrr(line, file)
    map,   line = read_map(line, file)

    return Results(precs=precs, recs=recs, mrr=mrr, map=map, perp=perp), line


def read_perplexity(line, file):
    '''
    Reads perplexity from the given file. The line is the current
    line read in, which may contain the perplexity
    :param line: the current line in the file
    :param file: the file to read from
    :return: the perplexity and the current line, in that order
    '''
    while line != "":
        line = line.strip()
        if line.startswith(PerplexityPrefix):
            return float(line[len(PerplexityPrefix):]), file.readline()
        if line.startswith(PrecisionTitle):
            return -1, line
        line = file.readline()

    return -1, line


def read_precs(line, file):
    while line != "":
        line = line.strip()
        if line.startswith(PrecisionTitle):
            return read_table(file)

        if line.startswith(RecallTitle):
            return new_group_k_dict(), line
        line = file.readline()

    return new_group_k_dict(), line


def read_recs(line, file):
    while line != "":
        line = line.strip()
        if line.startswith(RecallTitle):
            return read_table(file)

        if line.startswith(MrrPrefix):
            return new_group_k_dict(), line
        line = file.readline()

    return new_group_k_dict(), line


def read_table(file):
    # We assume the table is fully formed and correct
    result = new_group_k_dict()

    # The next two lines are headings
    file.readline()
    file.readline()

    # The next five lines are values
    for group in FileGroups:
        line = file.readline().strip()

        cols = line.split("|")
        for i,k in enumerate(TableTitles):
            result[group][k] = float(cols[i + 2].strip())

    # Sum up the averages
    roll_up_and_average(result, Range3_up, [Range3_5, Range5_10, Range10_up])
    roll_up_and_average(result, Range5_up, [Range5_10, Range10_up])

    return result, file.readline()


def roll_up_and_average(dic, dst, srcs):
    for src in srcs:
        dic[dst][DocCount] += dic[src][DocCount]
        for m in Ms:
            dic[dst][m] += dic[src][m] * dic[src][DocCount]
    for m in Ms:
        dic[dst][m] = dic[dst][m] / dic[dst][DocCount]


def read_mrr(line, file):
    while line != "":
        line = line.strip()
        if line.startswith(MrrPrefix):
            return float(line[len(MrrPrefix):]), file.readline()
        if line.startswith(MapPrefix):
            return -1, line
        line = file.readline()

    return -1, line

def read_map(line, file):
    while line != "":
        line = line.strip()
        if line.startswith(MapPrefix):
            return float(line[len(MapPrefix):]), file.readline()
        if line == "":
            return -1, line
        line = file.readline()

    return -1, line

def skip_to_non_empty_line(line, file):
    '''
    Read in lines from the file until we run out of lines or reach
    a non-empty line.

    Returns a line and the file object. The line is either non-empty
    or "", the latter indicating that we've reached the end of the file.
    '''

    if line == "": # End of file
        return "", file
    stripped = line.strip()
    if not (stripped == "" or stripped.startswith('#')): # Non-Empty, non-commented line
        return line, file

    return skip_to_non_empty_line(file.readline(), file)

def read_all_results(path):
    results = dict()
    with open (path, "r") as file:
        line = "\n"
        while line != "":
            line, file = skip_to_non_empty_line(file.readline(), file)
            algorDesc = Algorithm.from_file_entry(line)
            line, file = skip_to_non_empty_line(file.readline(), file)
            if line == "":
                raise ValueError("No results for " + str(algorDesc))

            algorResults, line = read_next_result_batch(line, file)
            results[algorDesc] = algorResults
            print ("Read in results for " + str(algorDesc))

    return results

def write_all_results(path, results_dict):
    '''
    Writes out results in a formula amendable to analysis by R
    :param path: the path to which the results should be written to
    :param results: a dictionary mapping Algorithm objects to
    Results object
    '''
    ColDelim=","
    RowDelim="\n"
    TextLeftQuote="\""
    TextRightQuote="\""
    TextQuoteEscape="\"" # Excel uses Pascal style quote-escaping

    def toNullable(value):
        return "" if value < 0.9 else str(value)

    def quoteText(value):
        # Use Pascal / Excel style quote escaping
        value = value.replace(TextLeftQuote,  TextQuoteEscape + TextLeftQuote)
        value = value.replace(TextRightQuote, TextQuoteEscape + TextRightQuote)
        value = TextLeftQuote + value + TextRightQuote
        return value

    with open (path, "w") as f:
        f.write(ColDelim.join (["Algor", "TopicCount", "TrainDesc", "MRR", "MAP", "Perplexity", "Group", "M", "DocCount", "Precision", "Recall"]))
        f.write(RowDelim)

        for algor, results in results_dict.items():
            for group in FullGroups:
                for m in Ms:
                    f.write(algor.to_delim_str(ColDelim, quoteText))
                    f.write(ColDelim)

                    f.write(str(results.mrr))
                    f.write(ColDelim)

                    f.write(str(results.map))
                    f.write(ColDelim)

                    f.write(toNullable(results.perp))
                    f.write(ColDelim)

                    f.write(quoteText(group))
                    f.write(ColDelim)

                    f.write(m)
                    f.write(ColDelim)

                    f.write(str(int(results.precs[group]["DocCount"])))
                    f.write(ColDelim)

                    f.write(str(results.precs[group][m]))
                    f.write(ColDelim)

                    f.write(str(results.recs[group][m]))
                    f.write(RowDelim)



def run(args):
    inPath  = args[0] if len(args) > 0 else DefaultInPath
    outPath = args[1] if len(args) > 1 else DefaultOutPath
    results = read_all_results(inPath)
    write_all_results(outPath, results)

if __name__ == '__main__':
    run(args=sys.argv[1:])