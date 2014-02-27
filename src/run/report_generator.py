'''
Given a directory full of output files,
generates a report for each one.

@author: bryanfeeney
'''
import os
import sys
import shutil
import argparse as ap
from sys import stderr
from string import Template
from re import match

from os.path import sep

CtmTemplateFileName = "CtmResultsSheet-Template.ipynb"
StmTemplateFileName = "StmResultsSheet-Template.ipynb"
TemplateDir = os.path.dirname (os.path.dirname (__file__)) + os.path.sep + 'notebooks'
CodeDir     = os.path.dirname(TemplateDir)
CtmOutFileNameFormat = r"ctm_%s_k_%d_fold_\d_\d{8}_\d{4}.pkl"
StmOutFileNameFormat = r"stm_yv_%s_k_%d_p_%d_fold_\d_\d{8}_\d{4}.pkl"

CtmReportFileFormat = "ctm_%s_k_%d.ipynb"
StmReportFileFormat = "stm_yv_%s_k_%d_p_%d.ipynb"

LatexTemplate = "article_nocode.tplx"
PdfConversionScript = "convert-to-pdf.sh"

ExpectedFoldCount = 5

# Names for the model
Bohning  = 'bohning'
Bouchard = 'bouchard'
Ctm      = 'ctm'
StmYv    = 'stm_yv'

implNames = {'ctm':    {'bouchard': 'ctm', 'bohning':'ctm_bohning'}, \
             'stm_yv': {'bouchard': 'stm_yv', 'bohning':'stm_yv_bohning'}}

def run(args):
    '''
    Parses the command-line arguments (excluding the application name portion). 
    Executes a report generation run accordingly.
    
    Returns the list of files created.
    '''
    
    #
    # Enumerate all possible arguments
    #
    parser = ap.ArgumentParser(description='Execute a report generation task.')
    parser.add_argument('--model', '-m', dest='model', metavar=' ', \
                    help='The type of mode to use, options are ctm_bouchard, ctm_bohning, stm_yv_bouchard, stm_yv_bohning')
    parser.add_argument('--topic-list', '-k', dest='topic_counts', metavar=' ', \
                    help='The comma-separated list of topic-counts')
    parser.add_argument('--lat-sizes', '-q', dest='latent_sizes', metavar=' ', \
                    help='The comma-separated list of latent sizes')
    parser.add_argument('--report-dir', '-o', dest='report_dir', metavar=' ', \
                    help='The directory in which the reports are placed.')
    parser.add_argument('--template-dir', '-t', dest='template_dir', metavar=' ', \
                    help='The directory containing the report templates, if you wish to specify an alternative to the built-ins')
    parser.add_argument('--output-dir', '-i', dest='output_files', metavar=' ', \
                    help='The directory containing the model outputs')
    
    
    #
    # Parse the arguments
    #
    print ("Args are : " + str(args))
    args = parser.parse_args(args)
    
    if args.model.endswith(Bohning):
        bounds = [Bohning]
        model  = args.model[:-len(Bohning)]
    elif args.model.endswith(Bouchard):
        bounds = [Bouchard]
        model  = args.model[:-len(Bouchard)]
    else:
        bounds = [Bouchard, Bohning]
        model = args.model
    
    
    topicCounts = [int(countStr) for countStr in args.topic_counts.split(',')]
    if args.latent_sizes is not None:
        latentSizes = [int(sizeStr) for sizeStr in args.latent_sizes.split(',')]
    
    #
    # Launch the report
    #
    if model == Ctm:
        generate_reports_ctm(bounds, topicCounts, args.output_files, args.report_dir, args.template_dir)
    elif model == StmYv:
        generate_reports_stm_yv(bounds, topicCounts, latentSizes, args.output_files, args.report_dir, args.template_dir)
    else:
        raise ValueError ("No such model " + model + " (derived from " + args.model + ")")
    
    

def _generate_report(fnameRegex, rawOutDir, reportFile, templateDir, modelType, bound):
    '''
    Generates a single report with the given configuration.
    
    Returns the number of folds
    '''
    # Load the files and check we have the expected number of outputs for a 
    # n-fold cross validation
    fnames = [fname for fname in os.listdir(rawOutDir) if match(fnameRegex, fname)]
    folds  = len(fnames)
    
    if folds == 0:
        stderr.write("No output to process for report " + reportFile)
        return 0
    elif folds < ExpectedFoldCount:
        stderr.write("Only " + str(len(fnames)) + " folds were written out for report " + reportFile)
    
    # Load the template and use it to create the report
    templateName = CtmTemplateFileName if modelType == 'ctm' else StmTemplateFileName
    with open(templateDir + sep + templateName, 'r') as f:
        templateStr = f.read()
    
    template = Template(templateStr)
    report = template.substitute( \
        bound = bound, \
        codePath = "'" + CodeDir + "'", \
        outFilesPrefix = "'" + rawOutDir + sep + "'", \
        outputFiles = ", ".join("outputPathPrefix + '" + fname + "'" for fname in fnames), \
        implName = implNames[modelType][bound])
    
    # Save the report
    with open(reportFile, 'w') as f:
        f.write(report)
    print("Wrote report to " + reportFile)
    
    return folds

def generate_reports_ctm (bounds, topicCounts, rawOutDir, reportDir, templateDir=None):
    '''
    Using raw outputs in the raw output directory, and
    report templates in the template directory (whose names
    are hardcoded), generates reports for every possible
    model configuration with output.
    
    We expect a certain naming convention to be used (see
    code) for the raw outputs
    
    These reports are configured, but not executed, so will
    have stale results in them. To execute them, load them
    in iPython and "Run All" or else run on the command-line
    using runipy
    
    Params:
    bounds      - the bounds tried, same syntax as main.py
    topicCounts - the numbers of topics tried, a list of ints
    rawOutDir   - contains the raw output, a pickle for every fold
    reportDir   - where the genereated reports are placed
    templateDir - where the report templates are located.
    
    Returns:
    A dictionary of report name folds processed. Folds set to
    zero if there was an error.
    '''
    if templateDir is None:
        templateDir = TemplateDir
    
    results = dict()
    for bound in bounds:
        for topicCount in topicCounts:
            reportFile = reportDir + sep + CtmReportFileFormat % (bound, topicCount)
            fnameRegex = CtmOutFileNameFormat % (bound, topicCount)
            
            foldCount = _generate_report(fnameRegex, rawOutDir, reportFile, templateDir, 'ctm', bound)
            results[reportFile] = foldCount
    
    _copyPdfConversionFiles(reportDir, templateDir)
    return results

def generate_reports_stm_yv (bounds, topicCounts, latentSizes, rawOutDir, reportDir, templateDir=None):
    '''
    Using raw outputs in the raw output directory, and
    report templates in the template directory (whose names
    are hardcoded), generates reports for every possible
    model configuration with output.
    
    We expect a certain naming convention to be used (see
    code) for the raw outputs
    
    These reports are configured, but not executed, so will
    have stale results in them. To execute them, load them
    in iPython and "Run All" or else run on the command-line
    using runipy
    
    Params:
    bounds      - the bounds tried, same syntax as main.py
    topicCounts - the numbers of topics tried, a list of ints
    latentSizes - the sizes of the latent spaces
    rawOutDir   - contains the raw output, a pickle for every fold
    reportDir   - where the genereated reports are placed
    templateDir - where the report templates are located.
    
    Returns:
    A dictionary of report name folds processed. Folds set to
    zero if there was an error.
    '''
    if templateDir is None:
        templateDir = TemplateDir
    
    results = dict()
    for bound in bounds:
        for topicCount in topicCounts:
            for latentSize in latentSizes:
                reportFile = reportDir + sep + StmReportFileFormat % (bound, topicCount, latentSize)
                fnameRegex = StmOutFileNameFormat % (bound, topicCount, latentSize)
                
                foldCount = _generate_report(fnameRegex, rawOutDir, reportFile, templateDir, 'stm_yv', bound)
                results[reportFile] = foldCount
    
    _copyPdfConversionFiles(reportDir, templateDir)
    return results


def _copyPdfConversionFiles(reportDir, templateDir):
    '''
    Copies across the PDF conversion script and any associated files which 
    are required for that conversion
    '''
    _copyIfAbsent(reportDir, templateDir, LatexTemplate)
    _copyIfAbsent(reportDir, templateDir, PdfConversionScript)


def _copyIfAbsent(dst, src, fname):
    '''
    Copies the file fname to the dst directory from the src directory if it 
    isn't already present
    '''
    
    if not os.path.exists(dst + sep + fname):
        shutil.copy (src + sep + fname, dst + sep + fname)


if __name__ == '__main__':
    run(args=sys.argv[1:])
