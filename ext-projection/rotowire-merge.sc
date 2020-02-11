#!/bin/fish
set FILES rotowire-*csv
set NEW_FILE rotowire_all.csv
for f in $FILES
         echo $f 1>&2
         if head -n 1 $f | grep -q "DFS POS"
             set f2 /tmp/rotowire-merge.tmp.csv
             cut -d, -f 4 --complement $f > $f2
         else
             set f2 $f
         end
         sed -e "s/^/$f,/" $f2
end | sed -n '1p ; /.*PLAYER.*/!p' > $NEW_FILE
