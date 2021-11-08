#!/usr/bin/env python
import os
import tarfile
from subprocess import call
import argparse

arg_parser = argparse.ArgumentParser(description='backup of data for pzs or rss on tapes')
arg_parser.add_argument('-r','--read',help='only read data from tape, no default', default='t')
arg_parser.add_argument('-e','--extract',help='only extract data from tape, no default', default='f')
#arg_parser.add_argument('-n','--number',help='how many tapes are inserted', default='8')

#if you only want to read the data set read='t' (True) and extract='f' (False)
#if you want to extract the data: set extract='t' True and think about the path where you want the data to be extracted

args = arg_parser.parse_args()

read=args.read
extract=args.extract
#n=args.number

if __name__ == "__main__":
#    for j in range(1,n):
#        os.system('mtx -f /dev/sg1 load '+j)
        call(['mt','-f','/dev/nst0','rewind'])
        
        if read=='t':    
            os.system('tar -tvf /dev/nst0')
            os.system('mt -f /dev/nst0 fsf 2')   
        
            for i in range(1,20):
                os.system('tar -tvf /dev/nst0')
                os.system('mt -f /dev/nst0 fsf 2')
        
            call(['mt','-f','/dev/nst0','eod'])
            os.system('mt -f /dev/nst0 bsfm 2')   
            os.system('tar -tvf /dev/nst0')
    
    
        if extract=='t':
            os.system('tar -b 1024 -xvf /dev/nst0')
            os.system('mt -f /dev/nst0 fsf 1')   
        
            for i in range(1,20):
                os.system('tar -b 1024 -xvf /dev/nst0')
                os.system('mt -f /dev/nst0 fsf 1') 
                os.system('tar -xvf /dev/nst0')      
                os.system('mt -f /dev/nst0 fsf 1')     
        
            call(['mt','-f','/dev/nst0','eod'])
            os.system('mt -f /dev/nst0 bsfm 2')   
            os.system('tar -b 1024 -xvf /dev/nst0')
            
        call(['mt','-f','/dev/nst0','rewind'])
#        os.system('mtx -f /dev/sg1 unload '+j)
    
###############################################################################
#nach Anmeldung auf lto5
#mtx -f /dev/sg1 load 1
#cd /vols/talos/home/kevin/copy_data/
#python /vols/talos/home/kevin/code_python/ms15_extract_backwards.py
#now you have to unload the tape and load the next tape with mtx -f /dev/sg1 unload 1 and mtx -f /dev/sg1 load 2
