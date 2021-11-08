# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:54:22 2016
last update 24.06.2016 14:00
@author: ohneiser
"""
from subprocess import call

class Tape(object):
    
    def __init__(self, dev_tape, dev_loader, log=None, execute=True):     
        self.dev_tape = dev_tape
        self.dev_loader = dev_loader
        self.execute = execute
        if log is not None:
            self.log = open(log,'a')
        else:
            self.log = None
        return
###############################################################################
        
    def load(self,i,dev):
        '''
        load a tape
        i : tape number
        '''
        cmd = 'mtx -f {dev} load {num}'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')
        if self.execute:
            call(['mtx','-f',self.dev_loader,'load',str(i)])
        
        return
###############################################################################        
 
    def unload(self,i,dev):
        '''
        unload a tape
        i : tape number
        '''
        cmd = 'mtx -f {dev} unload {num}'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')
        if self.execute:
            call(['mtx','-f',self.dev_loader,'unload',str(i)]) 
        return
###############################################################################
        
    def rewind(self, i,dev):
        '''
        wind a tape to the begin
        i : tape number
        '''
        cmd = 'mt -f /dev/nst0 rewind'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')
        if self.execute:
            call(['mt','-f',self.dev_tape,'rewind'])
        return
###############################################################################
        
    def bsfm(self, i,dev):
        '''
        wind a tape one file backwards
        i : tape number
        '''
        cmd = 'mt -f /dev/nst0 bsfm 2'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')   
        if self.execute:
            call(['mt','-f',self.dev_tape,'bsfm','2'])            
        return
###############################################################################
        
    def fsf(self, i,dev):
        '''
        wind a tape one file backwards
        i : tape number
        '''
        cmd = 'mt -f /dev/nst0 fsf 1'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')   
        if self.execute:
            call(['mt','-f',self.dev_tape,'fsf','1'])            
        return
###############################################################################        
        
    def eod(self, i,dev):
        '''
        wind to the end of a tape
        i : tape number
        '''
        cmd = 'mt -f /dev/nst0 eod'.format(dev=self.dev_loader,num=i)
        if self.log is not None:
            self.log.write(cmd+'\n')       
        if self.execute:
            call(['mt','-f',self.dev_tape,'eod'])
        return
###############################################################################

