'''
Created on 27 Feb 2014

@author: bryanfeeney
'''
import unittest

import run.report_generator as rep_gen

ResultsDir = '/Users/bryanfeeney/Desktop/nips-out/out'
ReportsDir = '/Users/bryanfeeney/Desktop/nips-out/reports'

class Test(unittest.TestCase):


    def testGenerationOnRealCtmOutput(self):
        cmdline = '' \
                + ' --model '          + 'stm_yv' \
                + ' --output-dir '     + ResultsDir \
                + ' --report-dir '     + ReportsDir \
                + ' --topic-list '     + '5,10,25' \
                + ' --lat-sizes '      + '5,10,25,50,75,100' \
                + ' --dataset '        + rep_gen.NIPS
        
        argv = cmdline.strip().split(' ')
        rep_gen.run(argv)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGenerationOnRealReport']
    unittest.main()