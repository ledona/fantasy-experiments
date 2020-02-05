#/bin/fish
set FILES rotogrinder-*csv
set NEW_FILE grinder_all.csv
echo file,player,salary,team,position,opp-team,ceil,floor,pts > $NEW_FILE
for f in $FILES
sed -e "s/^/$f,/" $f
end >> $NEW_FILE
