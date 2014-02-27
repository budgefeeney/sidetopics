'''
Created on 27 Feb 2014

@author: bryanfeeney
'''
import unittest

import run.report_generator as rep_gen

k5_results_dir = '/Users/bryanfeeney/Desktop/k5_results'

class Test(unittest.TestCase):


    def testGenerationOnRealCtmOutput(self):
        cmdline = '' \
                + ' --model '          + 'ctm' \
                + ' --topic-list '     + '5,10,25,50' \
                + ' --output-dir '     + k5_results_dir \
                + ' --report-dir '     + k5_results_dir
        
        argv = cmdline.strip().split(' ')
        rep_gen.run(argv)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGenerationOnRealReport']
    unittest.main()