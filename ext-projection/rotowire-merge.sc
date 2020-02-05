#/bin/fish
set FILES rotowire-*csv
set NEW_FILE rotowire_all.csv
for f in $FILES
sed -e "s/^/$f,/" $f
end | sed -n '1p ; /.*PLAYER.*/!p' > $NEW_FILE
