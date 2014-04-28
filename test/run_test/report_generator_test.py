'''
Created on 27 Feb 2014

@author: bryanfeeney
'''
import unittest

import run.report_generator as rep_gen

ResultsDir = '/Users/bryanfeeney/Desktop/tweets-dow/out'
ReportsDir = '/Users/bryanfeeney/Desktop/tweets-dow/reports'

class Test(unittest.TestCase):


    def testGenerationOnRealOutput(self):
        for algor in [ 'ctm', 'stm_yv' ]:
            cmdline = '' \
                + ' --model '          + algor \
                + ' --output-dir '     + ResultsDir \
                + ' --report-dir '     + ReportsDir \
                + ' --topic-list '     + '5,10,25,50,100,150,250' \
                + ' --lat-sizes '      + '5,10,25,50,75,100'
        
            argv = cmdline.strip().split(' ')
            rep_gen.run(argv)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGenerationOnRealReport']
    unittest.main()