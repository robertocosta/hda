git pull
echo " ">>jobs.txt
commit_n=$(git log --pretty=format:'%H' -n 1)
commit_d=$(git log --pretty=tformat:'%cd' -n 1)
commit_d=${commit_d//[ :]}
qsout=$(qsub jmain.job)
printf "%s,%s,https://gitlab.dei.unipd.it/costarob/hda/blob/%s/Models.py" ${commit_d:0:18} ${qsout:9:7} $commit_n >>jobs.txt
qstat
git add jobs.txt
git commit -m "jobs update"
git push
