# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:02:56 2016
last update 24.06.2016 15:00
@author: ohneiser
"""

import argparse
import data
import tape
import datetime
from subprocess import call
import os
import tarfile

arg_parser = argparse.ArgumentParser(description='backup of data for pzs or rss on tapes')
arg_parser.add_argument('-y','--year',help='year; NO DEFAULT VALUE')
arg_parser.add_argument('-m','--month',help='month; NO DEFAULT VALUE')
arg_parser.add_argument('-s','--service',help='service (pzs | rss); NO DEFAULT VALUE')
arg_parser.add_argument('-n','--num_months',help='number of months (max. 64); default=8', default=8)
arg_parser.add_argument('-d','--dev_loader',help='device; default=/dev/sg1', default='/dev/sg1')
arg_parser.add_argument('-t','--temp',help='Ort für temp Ordner; default=/vols/talos/home/kevin/temp/',default='/vols/talos/home/kevin/temp/')
arg_parser.add_argument('-c','--copypath',help='path, where to copy data from; default=/vols/altair/datasets/eumcst/', default='/vols/altair/datasets/eumcst/')
arg_parser.add_argument('-v','--version',help='version number; default=1.0', default='1.0')
arg_parser.add_argument('-z','--months_per_tape',help='how many months per tape (max. 8);  default=8', default=8)
arg_parser.add_argument('-p','--path',help='where to write log file;  default=/vols/talos/home/kevin/', default='/vols/talos/home/kevin/')
arg_parser.add_argument('-T','--dev_tape',help='default: /dev/nst0', default='/dev/nst0')
args = arg_parser.parse_args()

#for dev:  /vols/talos/home/kevin/temp

print args.month
print args.year
print args.service
print args.num_months
print args.copypath
print args.version
print args.months_per_tape
print args.path

if __name__ == "__main__":
    #create temporary folder (anywhere)
    os.mkdir('/vols/talos/home/kevin/temp')
    #parse the inserted arguments
    month = int(args.month)
    year  = int(args.year)
    service = args.service
    num_months = int(args.num_months)
    temp=args.temp
    copyfrompath=args.copypath
    version=args.version
    num_months_per_tape=int(args.months_per_tape)
    path=args.path    
    dev_loader=args.dev_loader
    dev_tape=args.dev_tape
    
    #open test.txt and inventory list
    open(path+'test.txt','w').close()  
    open(temp+'inventory_list.txt','w').close()
    itape=1
    #Routine for transition between december and january of the "new" year; important for dt
    j=0
    i=0
    b=[]
    c=[]
    datelist=[]
    b[i:i]=[month+j]
    c[i:i]=[year]
    if month+j>12:
        while (b[i]>12):
            c[i]=c[i]+1
            b[i]=b[i]-12
    datelist[i:i] = [datetime.datetime(c[i],b[i],1)]        
    dt=datetime.datetime(c[i],b[i],1)
      
    #calculate number of needed tapes
    num_tapes=num_months/8+1
    if (num_months%8==0):
        num_tapes=num_months/8    
    if (num_tapes<9):
        tapedev =tape.Tape("/dev/nst0","/dev/sg1",log=path+'test.txt')
        #loop for tapes (which tape number is loaded)
        for itape in range(num_tapes):
            if (j<num_months):
                i=1
                #load and rewind next tape and put identifier to the beginnig
                tapedev.load(itape+1,dev_loader)
                tapedev.rewind(itape+1,dev_tape)                
                fid=open(temp+'identifier.txt','w') 
                data.write_identifier(fid,version,temp)
                fid.close()         
                tape_dev = '/dev/nst0'
                blocksize = 10240
                f = tarfile.open(tape_dev, 'w|', bufsize=blocksize)
                f.add('/vols/talos/home/kevin/temp/identifier.txt',arcname='identifier.txt',recursive=True)
                f.close()
                #loop for the segmasks, data and inventory file that will be written on the inserted tape
                for i in range(num_months_per_tape):
                    if (j<num_months):
                        i=i+1
                        flog=open(path+'test.txt','a')
                        data.write_segmasks(flog,month,year,service,temp,copyfrompath,path,j,temp)         
                        data.write_data(flog,month,year,service,temp,copyfrompath,path,j,temp)         
                        tapedev.fsf(itape,dev_tape)
                        finv=open(temp+'inventory_list.txt','a')               
                        data.write_inventoryfile(finv,flog,month,year,j,service,temp,path)         
                        tapedev.fsf(itape,dev_tape)
                        finv.close()
                        flog.close()       
                        f = tarfile.open(tape_dev, 'w|', bufsize=blocksize)
                        f.add('/vols/talos/home/kevin/temp/inventory_list.txt',arcname='inventory_list.txt',recursive=True)
                        f.close()               
                        tapedev.bsfm(itape,dev_tape)
                        j=j+1
                #rewind the tape and unload it; increase the number of the tape,so the next tape can be inserted
                tapedev.rewind(itape+1,dev_tape) 
                tapedev.unload(itape+1,dev_loader)
                itape=itape+1
    else:
        print 'There are only 8 places for tapes available, so num_months has to be smaller than 65 (with 8 tapes per month)'
    #erase the temporary folder that has been created at the begin
    call(['rm','-rf','/vols/talos/home/kevin/temp/'])
###############################################################################
#this is the main programm
#after logging in onto lto5 write:
#python /vols/talos/home/kevin/code_python/ms15_backup.py -n 2 -y 2006 -m 6 -s pzs
#this writes the pzs-data from june and july 2006 onto the tape