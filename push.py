import os
import glob
import re

files = glob.glob ('*git*.sh')
for f in files:
    with open (f, 'rU') as infile, \
            open (f + '.sh', 'w', newline='\n') as outfile:
        outfile.writelines (infile.readlines ())
    e = 'chmod 755 ' + f + '.sh'
    print(e)
    os.system (e)
    os.system ('./' + f + '.sh')
    os.remove (f)
    os.remove (f + '.sh')

jobIDs=[]
datestrs=[]
try:
    tab = open ('job_table.txt', 'r')
    for l in tab.readlines():
        jobIDs.append (l.split (',')[0])
        datestrs.append (l.split (',')[1])
    tab.close()
except:
    print('From scratch')

files = glob.glob ('*.job.o*')
for fname in files:
    jobID = fname.split('.job.o')[1]
    if jobID not in jobIDs:
        r = open (fname, 'r')
        datestr = r.readline()
        m = re.search ('[0-9]*\n', datestr)
        if hasattr(m,'group'):
            jobIDs.append(jobID)
            datestrs.append(datestr)

tab = open('job_table.txt', 'w')
for i in range(len(jobIDs)):
    tab.write('{},{}'.format(jobIDs[i],datestrs[i]))

tab.close()
