# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:38:23 2016
last update 24.06.2016 15:00
@author: ohneiser
"""

import os
import datetime
import tarfile
from subprocess import call
import socket, uuid, getpass
###############################################################################
'''    
def iter_files(path):
#für das Überprüfen der Datenmenge auf der Festplatte
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
           files.append(os.path.join(root, filename))

return files
'''
###############################################################################

def write_identifier(fid,version,dev):    
#Create the identifier.txt and write the format, uid, creation time, version number, user and hostname 
    uid = uid=str(uuid.uuid4().get_hex().upper()[0:12])
    host = socket.gethostname()
    user = getpass.getuser()
    dt = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    fid.writelines('format=MSG_L15_BACKUP')    
    fid.writelines('\n')
    fid.writelines('uuid=' + uid)
    fid.writelines('\n')
    fid.writelines('creation_time=' + dt) 
    fid.writelines('\n')
    fid.writelines('version=' + version) 
    fid.writelines('\n')   
    fid.writelines('user=' + user) 
    fid.writelines('\n')
    fid.writelines('host=' + host)
    
############################################################################### 
    
def write_segmasks(flog,month,year,service,dev,copyfrompath,path,j,temp):       
#Routine for transition between december and january of the "new" year; important for dt
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

#Write segmask data onto the tape
    tape_dev = '/dev/nst0'
    blocksize = 10240 
    
    dir_fmt = '/vols/altair/datasets/eumcst/msevi_%s/meta/segmasks/%s/' % (service,dt.strftime('%Y/%m'))
    id_file = temp+'identifier.txt'

    f = open(id_file,'r')
    txt = f.read()
    f.close()
    print txt

    tf = open(tape_dev,'w')
    tf.write(txt)
    tf.close()

    dir = dir_fmt
    f = tarfile.open(tape_dev, 'w|', bufsize=blocksize)
    print "Storing: ", dir
   
   #key line:
    f.add(dir,arcname='segmasks/%s/' % (dt.strftime('%Y/%m')),recursive=True)     
    f.close()
    
#only for Logfile test.txt (documentation of important steps )          
    flog.writelines('segment mask data actual')
    flog.writelines('\n')
############################################################################### 
    
def write_data(flog,month,year,service,dev,copyfrompath,path,j,temp): 
    """
    function to write inventory file
    INPUT:
    flog: for log data
    month: of the date the data is written
    year: of the date the data is written
    service: you use
    dev: /dev/nst0
    copyfrompath: default=/vols/altair/datasets/eumcst/
    path: default=/vols/talos/home/kevin/
    j:carryover
    temp: temporary folder at /vols/talos/home/kevin/temp/
    """        
#Routine for transition between december and january of the "new" year; important for dt
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
    
#write the 5|15 min data onto the tape
    tape_dev = '/dev/nst0'
    blocksize = 10240 
    
    dir_fmt = '/vols/altair/datasets/eumcst/msevi_%s/l15_hrit/%s/' % (service,dt.strftime('%Y/%m'))
    id_file = temp+'identifier.txt'

    f = open(id_file,'r')
    txt = f.read()
    f.close()

    tf = open(tape_dev,'w')
    tf.write(txt)
    tf.close()

    dir = dir_fmt
    f = tarfile.open(tape_dev, 'w|', bufsize=blocksize)
    print "Storing: ", dir
   #key line
    f.add(dir,arcname='data/%s/' % (dt.strftime('%Y/%m')),recursive=True)
    f.close()
    
    '''
    directory='/vols/altair/datasets/eumcst/msevi_pzs/l15_hrit/2006/06/'
    
    files = iter_files(directory)
    dir_size = 0

    for file in files:
        dir_size += os.path.getsize(file)

    print dir_size
    
    #open(inventory_list.txt,'a')
    #writelines(dir_size)
    #close(inventory_list.txt)
    '''
    
#only for Logfile test.txt (documentation of important steps)          
    flog.writelines('data actual')
    flog.writelines('\n')

###############################################################################
      
def write_inventoryfile(finv,flog,month,year,j,service,dev,path):    
    """
    function to write inventory file
    INPUT:
    finv: for inventory list
    flog: for log data
    month: of the date the data is written
    year: of the date the data is written
    j:carryover
    service: you use
    dev: /dev/nst0
    path: default=/vols/talos/home/kevin/
    """  
#Routine for transition between december and january of the "new" year; important for dt
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
    
#write saved content and new content into the inventory list
    finv.write('\n')
    finv.writelines(dt.strftime('%Y-%m'))
    finv.write('\t')
    finv.write(service)
    finv.write('\t')        
    finv.close()

#only for Logfile test.txt (documentation of important steps)  
    flog.writelines('inventory list is actual for ')
    flog.writelines(dt.strftime('%Y-%m'))
    flog.writelines('\n')
###############################################################################

#after logging in onto lto5 write:
#python /vols/talos/home/kevin/code_python/ms15_backup.py -n 2 -y 2006 -m 6 -s pzs
#this writes the pzs-data from june and july 2006 onto the tape





'''
stand in write segmasks:
#stats in einem tar-file
###################    call(['tar','-cvf','/dev/nst0',copyfrompath+'msevi_%s/meta/segmasks/%s/' % (service,dt.strftime('%Y/%m'))]) 

       
#kopieren der 5|15-minütigen Einzeldaten des rss|pzs - auskommentiert wegen großer Datenmengen
###################    call(['tar','-cvf','/dev/nst0',copyfrompath+'msevi_%s/l15_hrit/%s/' % (service,dt.strftime('%Y/%m'))]) 












'''