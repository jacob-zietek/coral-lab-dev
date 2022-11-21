cd ../data/haas-images
ffmpeg -r 1 -i ../11-17-22-haas-hallway-cut.MOV -r 1 "$filename%03d.png"
#find . -type f -print0 | sort -zR | tail -zn +1001 | xargs -0 rm